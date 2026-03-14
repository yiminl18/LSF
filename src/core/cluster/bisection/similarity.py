"""core.cluster.bisection.similarity -- Similarity helpers for the canonical fused pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import ot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm import tqdm

from core.cluster.common.paths import (
    BISECTION_SIM_PATH,
    OT_ALPHA,
    OT_EPSILON,
    OT_TAU,
)

logger = logging.getLogger(__name__)


def build_heading_tfidf_similarity(
    doc_ids: np.ndarray,
    rep: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the heading-text TF-IDF similarity matrix.
    
    Captures content similarity by vectorizing the bag-of-words representation 
    of all section headers within each document.
    """
    heading_docs = []
    for doc_id in doc_ids:
        texts = [node["text"] for node in rep[doc_id]["nodes"] if node.get("text")]
        heading_docs.append(" ".join(texts).lower())

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(heading_docs)

    similarity = cosine_similarity(tfidf_matrix)
    np.clip(similarity, 0.0, 1.0, out=similarity)

    logger.info(
        "    TF-IDF: %s features, mean sim=%.3f",
        tfidf_matrix.shape[1],
        similarity[np.triu_indices(len(doc_ids), k=1)].mean(),
    )
    return similarity, tfidf_matrix


def build_tree_shape_similarity(
    doc_ids: np.ndarray,
    rep: dict,
) -> np.ndarray:
    """Build the tree-shape similarity matrix used by the fused pipeline.
    
    Captures structural similarity by extracting topological features 
    (depth histograms, branching factors, leaf ratios) and applying 
    an RBF kernel to the feature matrix.
    """
    max_depth_global = max(max(node["depth"] for node in rep[doc_id]["nodes"]) for doc_id in doc_ids)
    n_levels = max_depth_global + 1

    features = []
    for doc_id in doc_ids:
        nodes = rep[doc_id]["nodes"]
        depths = [node["depth"] for node in nodes]
        n_nodes = len(nodes)

        level_hist = np.zeros(n_levels, dtype=float)
        for depth in depths:
            level_hist[depth] += 1
        if n_nodes > 0:
            level_hist /= n_nodes

        max_depth = max(depths)
        leaf_count = sum(1 for depth in depths if depth == max_depth)
        internal_count = n_nodes - leaf_count
        leaf_ratio = leaf_count / n_nodes if n_nodes > 0 else 0.0
        branching = n_nodes / internal_count if internal_count > 0 else 1.0

        depth_probs = level_hist[level_hist > 0]
        entropy = float(-np.sum(depth_probs * np.log2(depth_probs)))

        features.append(np.concatenate([level_hist, [leaf_ratio, branching, entropy]]))

    feature_matrix = np.array(features)
    similarity = rbf_kernel(feature_matrix, gamma=1.0 / feature_matrix.shape[1])

    logger.info(
        "    Tree shape: %s features, mean sim=%.3f",
        feature_matrix.shape[1],
        similarity[np.triu_indices(len(doc_ids), k=1)].mean(),
    )
    return similarity


def _compute_ot_score(
    X: np.ndarray,
    Y: np.ndarray,
    depths_x: np.ndarray,
    depths_y: np.ndarray,
) -> float:
    """Compute hierarchical Optimal Transport (OT) score between two documents.
    
    Measures semantic similarity while penalizing structural depth mismatches, 
    effectively matching heading sequences based on both content and hierarchy.
    """
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    norms_x = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    norms_y = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-8)
    cos_sim = (X / norms_x) @ (Y / norms_y).T
    base_cost = 1.0 - np.clip(cos_sim, -1.0, 1.0)

    depth_diff = np.abs(depths_x[:, None] - depths_y[None, :])
    penalized_cost = base_cost * np.exp(OT_ALPHA * depth_diff**2)

    dummy_cost = max(float(base_cost.mean()), 1e-6)
    m, n = len(X), len(Y)
    extended = np.full((m + 1, n + 1), dummy_cost, dtype=np.float64)
    extended[:m, :n] = penalized_cost
    extended[m, n] = 0.0

    a = np.ones(m + 1, dtype=np.float64) / (m + 1)
    b = np.ones(n + 1, dtype=np.float64) / (n + 1)

    transport = ot.unbalanced.sinkhorn_unbalanced(
        a,
        b,
        extended,
        reg=OT_EPSILON,
        reg_m=OT_TAU,
        numItermax=200,
        stopThr=1e-6,
    )

    real_transport = transport[:m, :n]
    real_mass = float(real_transport.sum())
    if real_mass < 1e-12:
        return 0.0

    affinity_quality = float((real_transport * (1.0 - base_cost)).sum()) / real_mass
    marginal_a = float(transport[:m, :].sum())
    marginal_b = float(transport[:, :n].sum())
    coverage = real_mass / max(min(marginal_a, marginal_b), 1e-12)
    return float(np.clip(affinity_quality * min(coverage, 1.0), 0.0, 1.0))


def build_semantic_similarity_matrix(
    representations: dict[str, dict[str, Any]],
    valid_doc_ids: list[str],
    force: bool = False,
) -> np.ndarray:
    """Build the full-corpus semantic OT similarity matrix.
    
    Computes pairwise Optimal Transport distances between documents based on 
    their heading embeddings and structural depths.
    """
    if BISECTION_SIM_PATH.exists() and not force:
        return np.load(BISECTION_SIM_PATH)

    views = {}
    for doc_id in valid_doc_ids:
        rep = representations[doc_id]
        mask = rep["emb_mask"]
        views[doc_id] = (
            rep["semantic"][mask],
            np.asarray([node["depth"] for node in rep["nodes"]], dtype=float)[mask],
        )

    n_docs = len(valid_doc_ids)
    similarity = np.eye(n_docs, dtype=np.float64)
    total_pairs = n_docs * (n_docs - 1) // 2
    with tqdm(total=total_pairs, desc="Semantic OT") as progress:
        for i in range(n_docs):
            X, depths_x = views[valid_doc_ids[i]]
            for j in range(i + 1, n_docs):
                Y, depths_y = views[valid_doc_ids[j]]
                if len(X) > 0 and len(Y) > 0:
                    score = _compute_ot_score(X, Y, depths_x, depths_y)
                    similarity[i, j] = similarity[j, i] = score
                progress.update(1)

    BISECTION_SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(BISECTION_SIM_PATH, similarity)
    logger.info("Wrote semantic OT matrix %s to %s", similarity.shape, BISECTION_SIM_PATH)
    return similarity


__all__ = [
    "build_heading_tfidf_similarity",
    "build_tree_shape_similarity",
    "build_semantic_similarity_matrix",
]
