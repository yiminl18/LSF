#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Model Module

Evaluates trained ranking models using test splits.
Reads test splits from EXPERIMENT directory.
Reads models from EXPERIMENT directory.
Reads processing/embeddings from SHARED directory.
Writes results to EXPERIMENT directory.

Usage:
    python -m core.pipeline.evaluate_model --dataset pdfs --model_config xgb-sem-struc-v7 --parser docling
"""

import argparse
import json
import math
import sys
import os
import gc
import time
import warnings
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, replace

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.utils.io import load_labels
from core.utils.parallel import run_pool_with_progress
from core.ml.config import (
    MODEL_TYPE_TO_MODE,
    ML_MODEL_TYPES,
    DEFAULT_MODEL_TYPES,
    DEFAULT_SEEDS,
)
from core.ml.base import BaseClassifier
from core.ml.factory import load_classifier
from core.ml.features import RETAINED_MODE, extract_ml_features
from core.doc.feature_extract import (
    HeaderNode,
    DocumentContext,
    iter_section_headers,
    build_document_context,
)
from core.embed.embeddings import (
    load_document_embeddings,
    get_query_embedding,
    cosine_sim,
    cosine_sim_batch,
    build_header_embedding_matrix,
    get_model_name_for_provider,
)

import numpy as np
import torch

# Supported embedding providers
EMBED_PROVIDERS = [
    "openai",
    "azure",
    "openrouter",
]


def _processing_json_path(processing_dir: Path, doc_id: str) -> Path:
    return processing_dir / f"{doc_id}_reconstructed.json"


# ========== Global mutable state (process-level, not thread-safe) ==========
# _TEMP_CACHE_DIR: set by _worker_init in child processes, pointing to a temp dir created by the main process
# Stores an embeddings subset needed for this run, avoiding full data loading per worker
_TEMP_CACHE_DIR: Optional[Path] = None

# _RAW_DOC_CACHE: LRU-style document cache within each worker subprocess
# Caches (header_list, context, embeddings), indexed by doc_id
# Each subprocess maintains its own independent cache
_RAW_DOC_CACHE: Dict[
    str, Tuple[List[HeaderNode], DocumentContext, Dict[str, List[float]]]
] = {}
_MAX_DOC_CACHE_SIZE = 20


def _ensure_doc_cache_space():
    if len(_RAW_DOC_CACHE) >= _MAX_DOC_CACHE_SIZE:
        try:
            first_key = next(iter(_RAW_DOC_CACHE))
            del _RAW_DOC_CACHE[first_key]
        except StopIteration:
            pass


class EnsembleModel:
    """Ensemble of models trained with different seeds."""

    def __init__(self, models: List[BaseClassifier]):
        self.models = models
        self.feature_names = models[0].feature_names if models else None

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        preds = np.array([m.predict(X, **kwargs) for m in self.models])
        return preds.mean(axis=0)


def _load_cluster_ensemble(
    config: str,
    cluster_id: int,
    q_idx: int,
    cluster_models_dir: Path,
    xgb_device: str = "cpu",
    nn_device: Optional[str] = None,
) -> Optional[EnsembleModel]:
    """Load cluster model ensemble (multiple seeds)."""
    config_dir = cluster_models_dir / config
    model_files = sorted(config_dir.glob(f"c{cluster_id}_q{q_idx}_seed*.json"))
    model_files = [f for f in model_files if not f.name.endswith(".meta.json")]
    if not model_files:
        return None
    clfs = [load_classifier(str(p), device=nn_device) for p in model_files]
    for clf in clfs:
        _apply_xgb_device_override(clf, xgb_device)
    return EnsembleModel(clfs)


def _resolve_xgb_device(device_arg: str) -> str:
    """Resolve XGBoost evaluation device."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _resolve_mp_start_method(xgb_device: str, nn_device: str) -> Optional[str]:
    """Force spawn on CUDA paths to avoid CUDA re-initialization in forked subprocesses."""
    nn_uses_cuda = nn_device == "cuda" or (
        nn_device == "auto" and torch.cuda.is_available()
    )
    if xgb_device == "cuda" or nn_uses_cuda:
        return "spawn"
    return None


def _apply_xgb_device_override(model: BaseClassifier, xgb_device: str) -> None:
    """Override inference device parameters for xgb/lm models."""
    try:
        classifier_type = model.get_classifier_type()
    except Exception:
        classifier_type = ""
    if classifier_type not in {"xgb", "lm"}:
        return

    if hasattr(model, "params") and isinstance(model.params, dict):
        model.params["device"] = xgb_device
        if xgb_device == "cuda":
            model.params["tree_method"] = "gpu_hist"
            model.params["predictor"] = "gpu_predictor"
        else:
            model.params["predictor"] = "cpu_predictor"
            model.params.pop("tree_method", None)

    booster = getattr(model, "model", None)
    if booster is not None and hasattr(booster, "set_param"):
        if xgb_device == "cuda":
            booster.set_param(
                {
                    "device": "cuda",
                    "predictor": "gpu_predictor",
                    "tree_method": "hist",
                }
            )
        else:
            booster.set_param({"device": "cpu", "predictor": "cpu_predictor"})


# ── Score Aggregation ──

SCORE_AGG_METHODS = ("softmax", "top2_mean")


def _collect_score(scores: Dict[int, list], idx: int, pred: float) -> None:
    """Collect scores into a list (replaces max-pool, used for adaptive aggregation)."""
    scores.setdefault(idx, []).append(float(pred))


def _aggregate_scores(
    collected: Dict[int, list], method: str = "softmax", softmax_alpha: float = 5.0
) -> Dict[int, float]:
    """Aggregate collected score lists into a single score.

    method:
      - "softmax": log-sum-exp smooth approximation of max
      - "top2_mean": mean of the top two sources, reducing single-source noise
    """
    result: Dict[int, float] = {}
    for idx, vals in collected.items():
        if method == "softmax":
            arr = np.array(vals)
            # log-sum-exp: higher alpha approaches max
            shifted = softmax_alpha * arr
            result[idx] = float(
                np.log(np.sum(np.exp(shifted - shifted.max()))) / softmax_alpha
                + shifted.max() / softmax_alpha
            )
        elif method == "top2_mean":
            sorted_vals = sorted(vals, reverse=True)
            result[idx] = sum(sorted_vals[:2]) / min(len(sorted_vals), 2)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    return result


# Exported for analysis scripts (same reference as core.utils.io.load_labels)
load_test_labels = load_labels


def _temp_npz_path(temp_dir: Path, doc_id: str) -> Path:
    """Temp embeddings cache path (npz)."""
    return temp_dir / f"{doc_id}.npz"


def _save_temp_embeddings_npz(path: Path, emb_dict: Dict[str, Any]) -> None:
    """Save a temporary embeddings subset as npz (float32)."""
    keys = list(emb_dict.keys())
    if keys:
        values = np.array(
            [np.asarray(emb_dict[k], dtype=np.float32) for k in keys],
            dtype=np.float32,
        )
    else:
        values = np.empty((0, 0), dtype=np.float32)
    np.savez(path, keys=np.array(keys, dtype=object), values=values)


def _load_temp_embeddings_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load temporary npz embeddings."""
    data = np.load(path, allow_pickle=True)
    keys = data["keys"]
    values = data["values"]
    return {str(k): values[i] for i, k in enumerate(keys)}


def _sync_temp_cache(
    needed_keys: Dict[str, set],  # doc_id -> set of combined_text keys
    embeddings_dir: Path,
    processing_dir: Path,
    temp_dir: Path,
    provider: str,
) -> Tuple[int, int, int]:
    """
    Sync temporary cache: only cache the embeddings entries that are needed.
    Returns (docs_updated, keys_added, keys_removed) counts.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Clean up legacy json temp cache to avoid stale old-format files in the directory
    for legacy_json in temp_dir.glob("*.json"):
        legacy_json.unlink(missing_ok=True)

    # Current doc_ids in cache
    cached_ids = {f.stem for f in temp_dir.glob("*.npz")}
    needed_doc_ids = set(needed_keys.keys())

    # Remove documents no longer needed
    for doc_id in cached_ids - needed_doc_ids:
        _temp_npz_path(temp_dir, doc_id).unlink(missing_ok=True)

    docs_updated = 0
    keys_added = 0
    keys_removed = 0

    for doc_id, needed_text_keys in needed_keys.items():
        cache_path = _temp_npz_path(temp_dir, doc_id)

        # Load existing cache
        cached_emb: Dict[str, np.ndarray] = {}
        if cache_path.exists():
            cached_emb = _load_temp_embeddings_npz(cache_path)

        cached_keys = set(cached_emb.keys())
        to_add_keys = needed_text_keys - cached_keys
        to_remove_keys = cached_keys - needed_text_keys

        # Check if update is needed
        if to_add_keys or to_remove_keys:
            # Remove unneeded keys
            for k in to_remove_keys:
                cached_emb.pop(k, None)
            keys_removed += len(to_remove_keys)

            # Add missing keys (load from full embeddings)
            if to_add_keys:
                emb_load_path = _processing_json_path(processing_dir, doc_id)
                if emb_load_path.exists():
                    full_emb, _ = load_document_embeddings(
                        str(emb_load_path), str(embeddings_dir), provider=provider
                    )
                    for k in to_add_keys:
                        if k in full_emb:
                            cached_emb[k] = np.asarray(full_emb[k], dtype=np.float32)
                            keys_added += 1

            # Write updated cache (npz)
            _save_temp_embeddings_npz(cache_path, cached_emb)
            docs_updated += 1

    return docs_updated, keys_added, keys_removed


def _load_embeddings_from_temp(
    doc_id: str, temp_dir: Path
) -> Optional[Dict[str, np.ndarray]]:
    """Load embeddings for a single document from the temp cache (npz)."""
    cache_path = _temp_npz_path(temp_dir, doc_id)
    if cache_path.exists():
        return _load_temp_embeddings_npz(cache_path)
    return None


def _worker_init(temp_dir_str: str):
    """Worker initialization: set the temp cache directory."""
    global _TEMP_CACHE_DIR
    _TEMP_CACHE_DIR = Path(temp_dir_str)


def _collect_needed_keys(
    limit: int,
    splits_dir: Path,
    processing_dir: Path,
    max_target_docs: Optional[int] = None,
    q_indices: Optional[List[int]] = None,
) -> Dict[str, set]:
    """
    Collect embedding keys needed for this evaluation run.
    - Source docs (train): only provenance headers are needed
    - Target docs (test): all headers are needed
    Returns {doc_id: set(combined_text keys)}
    """
    needed_keys: Dict[str, set] = {}
    # Cache loaded header lists to avoid redundant JSON reads for the same document
    _header_cache: Dict[str, Optional[list]] = {}

    def _get_headers(doc_id: str) -> Optional[list]:
        if doc_id in _header_cache:
            return _header_cache[doc_id]
        load_path = _processing_json_path(processing_dir, doc_id)
        if not load_path.exists():
            _header_cache[doc_id] = None
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        header_list = iter_section_headers(data)
        _header_cache[doc_id] = header_list if header_list else None
        return _header_cache[doc_id]

    for q_idx in (q_indices if q_indices is not None else list(range(limit))):
        train_labels = load_labels(splits_dir / f"q{q_idx}_train_labels.json")
        test_labels = load_labels(splits_dir / f"q{q_idx}_test_labels.json")
        if max_target_docs is not None and max_target_docs > 0:
            test_labels = test_labels[:max_target_docs]

        # Process train docs (source): only provenance headers needed
        for label in train_labels:
            doc_id = label["doc_name"]
            header_list = _get_headers(doc_id)
            if not header_list:
                continue

            if doc_id not in needed_keys:
                needed_keys[doc_id] = set()

            # Resolve provenance -- header_idx is a texts[] global index, needs mapping to header_list position
            idx_to_pos = {h.idx_in_texts: i for i, h in enumerate(header_list)}
            prov_positions = set()

            nodes = label.get("possible_provenance_nodes", [])
            for node in nodes:
                label_idx = node.get("header_idx")
                if label_idx is not None:
                    pos = idx_to_pos.get(label_idx)
                    if pos is not None:
                        prov_positions.add(pos)

            # Add provenance headers' combined_text + processing_path
            for pos in prov_positions:
                h = header_list[pos]
                needed_keys[doc_id].add(h.combined_text)
                if h.processing_path:
                    needed_keys[doc_id].add(h.processing_path)

        # Process test docs (target): all headers needed
        for label in test_labels:
            doc_id = label["doc_name"]
            header_list = _get_headers(doc_id)
            if not header_list:
                continue

            if doc_id not in needed_keys:
                needed_keys[doc_id] = set()

            # Add all headers' combined_text + processing_path
            for h in header_list:
                needed_keys[doc_id].add(h.combined_text)
                if h.processing_path:
                    needed_keys[doc_id].add(h.processing_path)

    return needed_keys


@dataclass
class DocData:
    doc_id: str
    header_list: List[HeaderNode]
    context: DocumentContext
    embeddings: Dict[str, List[float]]
    prov_indices: set[int]


def load_doc_data(
    label_item: Dict[str, Any],
    processing_dir: Path,
    embeddings_dir: Path,
    provider: str = "openrouter",
) -> Optional[DocData]:
    doc_id = label_item["doc_name"]

    # 1. Check LRU Cache for heavy data
    if doc_id in _RAW_DOC_CACHE:
        # Move to end (LRU)
        header_list, context, embeddings = _RAW_DOC_CACHE.pop(doc_id)
        _RAW_DOC_CACHE[doc_id] = (header_list, context, embeddings)
    else:
        # Cache Miss: Load from disk
        load_path = _processing_json_path(processing_dir, doc_id)

        if not load_path.exists():
            return None

        with open(load_path, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

        header_list = iter_section_headers(merged_data)
        if not header_list:
            return None

        context = build_document_context(header_list)

        # Prefer loading embeddings from temp cache (smaller files, faster loading)
        embeddings = None
        if _TEMP_CACHE_DIR is not None:
            embeddings = _load_embeddings_from_temp(doc_id, _TEMP_CACHE_DIR)

        # Fallback: load from full embeddings directory
        if embeddings is None:
            embeddings, _ = load_document_embeddings(
                str(load_path), str(embeddings_dir), provider=provider
            )

        # Save to Cache
        _ensure_doc_cache_space()
        _RAW_DOC_CACHE[doc_id] = (header_list, context, embeddings)

    # 2. Resolve Provenance Indices (Always re-compute as it depends on label_item)
    prov_indices = set()
    text_to_list = {h.idx_in_texts: i for i, h in enumerate(header_list)}

    # possible_provenance_nodes -> header_list position
    nodes = label_item.get("possible_provenance_nodes", [])
    for node in nodes:
        t_idx = node.get("header_idx")
        if t_idx is not None:
            pos = text_to_list.get(t_idx)
            if pos is not None:
                prov_indices.add(pos)

    if not prov_indices:
        return None

    return DocData(
        doc_id=doc_id,
        header_list=header_list,
        context=context,
        embeddings=embeddings,
        prov_indices=prov_indices,
    )


def _compute_ndcg(ranks: List[Tuple[int, int]], k: int) -> float:
    """Compute NDCG@K (per-document average, binary relevance with single positive).
    Each (rank, _) represents the best provenance rank for one document.
    Per-doc: IDCG=1 (1 relevant at pos 1), DCG=1/log2(rank+1) if rank<=k else 0.
    """
    if not ranks:
        return 0.0
    total = 0.0
    for r, _ in ranks:
        if r <= k:
            total += 1.0 / math.log2(r + 1)  # IDCG = 1/log2(2) = 1.0
    return total / len(ranks)


def compute_metrics(ranks: List[Tuple[int, int]]) -> Dict[str, float]:
    if not ranks:
        return {
            "mrr": 0,
            "recall_1": 0,
            "recall_5": 0,
            "recall_10": 0,
            "ndcg_5": 0,
            "ndcg_10": 0,
            "mean_rank": 0,
        }

    mrr = sum(1.0 / r for r, _ in ranks) / len(ranks)
    recall_1 = sum(1 for r, _ in ranks if r <= 1) / len(ranks)
    recall_5 = sum(1 for r, _ in ranks if r <= 5) / len(ranks)
    recall_10 = sum(1 for r, _ in ranks if r <= 10) / len(ranks)
    mean_rank = sum(r for r, _ in ranks) / len(ranks)

    return {
        "mrr": mrr,
        "recall_1": recall_1,
        "recall_5": recall_5,
        "recall_10": recall_10,
        "ndcg_5": _compute_ndcg(ranks, 5),
        "ndcg_10": _compute_ndcg(ranks, 10),
        "mean_rank": mean_rank,
    }


def evaluate_target_doc(
    target: DocData,
    source_docs: List[DocData],
    query_embedding: List[float],
    query_text: str,
    models: List[Tuple[str, int, EnsembleModel]],
    source_labels: List[Dict[str, Any]],
    score_agg: str = "softmax",
    softmax_alpha: float = 5.0,
    return_raw_scores: bool = False,
) -> Dict[str, Any]:
    unsupported_modes = {mode for _, mode, _ in models if mode != RETAINED_MODE}
    if unsupported_modes:
        raise ValueError(
            f"Artifact only supports mode {RETAINED_MODE}, got modes {sorted(unsupported_modes)}"
        )

    n_target = len(target.header_list)
    q_vec = np.asarray(query_embedding, dtype=np.float64)
    dim = len(query_embedding)

    # ── Batch pre-compute target embedding matrix + query similarities ──
    target_emb_matrix, target_emb_valid, target_path_matrix, target_path_valid = (
        build_header_embedding_matrix(target.header_list, target.embeddings, dim)
    )

    target_query_sims = cosine_sim_batch(target_emb_matrix, q_vec)
    target_query_sims[~target_emb_valid] = 0.0

    target_path_query_sims = cosine_sim_batch(target_path_matrix, q_vec)
    target_path_query_sims[~target_path_valid] = 0.0

    from core.ml.lexical import (
        BM25Scorer,
        char3_jaccard,
        fuzzy_ratio,
        token_jaccard,
        token_precision,
        token_recall,
    )

    q_text = query_text or ""
    target_q_b_tok_jaccard = np.asarray(
        [token_jaccard(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_q_b_tok_recall = np.asarray(
        [token_recall(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_q_b_tok_precision = np.asarray(
        [token_precision(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_q_b_char3_jaccard = np.asarray(
        [char3_jaccard(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_q_b_fuzz_ratio = np.asarray(
        [fuzzy_ratio(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_q_path_b_tok_jaccard = np.asarray(
        [token_jaccard(q_text, h.processing_path or "") for h in target.header_list],
        dtype=np.float32,
    )

    target_bm25 = BM25Scorer([h.combined_text for h in target.header_list])
    target_bm25_header_scores = np.asarray(
        [target_bm25.score(q_text, h.text or "") for h in target.header_list],
        dtype=np.float32,
    )
    target_bm25_combined_scores = np.asarray(
        [target_bm25.score(q_text, h.combined_text or "") for h in target.header_list],
        dtype=np.float32,
    )

    rag_scores = [(i, float(target_query_sims[i])) for i in range(n_target)]

    feature_rows: List[Dict[str, float]] = []
    feature_target_indices: List[int] = []

    # Index source_labels for fast lookup (O(n) -> O(1))
    _source_label_map = {label["doc_name"]: label for label in source_labels}

    for source in source_docs:
        s_label = _source_label_map.get(source.doc_id)
        refined_list = s_label.get("refined_provenance") if s_label else None
        # Pre-build refined_list index
        refined_map: Dict[int, dict] = {}
        if refined_list:
            idx_to_pos = {h.idx_in_texts: i for i, h in enumerate(source.header_list)}
            for item in refined_list:
                t_idx = item.get("header_idx")
                if t_idx is None:
                    continue
                pos = idx_to_pos.get(t_idx)
                if pos is None and 0 <= t_idx < len(source.header_list):
                    pos = t_idx
                if pos is not None:
                    refined_map[pos] = item

        for idx_s in source.prov_indices:
            header_s_raw = source.header_list[idx_s]

            # Refined header (use dict lookup instead of linear scan)
            header_s_refined = header_s_raw
            if refined_map:
                refined_item = refined_map.get(idx_s)
                if refined_item and "visuals" in refined_item:
                    header_s_refined = replace(
                        header_s_raw,
                        font_size=refined_item["visuals"].get(
                            "font_size", header_s_raw.font_size
                        ),
                        is_bold=refined_item["visuals"].get(
                            "is_bold", header_s_raw.is_bold
                        ),
                    )

            # ── Pre-compute source header semantic similarity (invariant across all targets) ──
            a_emb_raw = source.embeddings.get(header_s_raw.combined_text)
            a_vec = (
                np.asarray(a_emb_raw, dtype=np.float64)
                if a_emb_raw is not None
                else None
            )
            pre_sim_a = float(cosine_sim(a_vec, q_vec)) if a_vec is not None else 0.0

            path_a = header_s_raw.processing_path
            emb_path_a = source.embeddings.get(path_a) if path_a else None
            pre_sim_path_a = (
                float(
                    cosine_sim(
                        np.asarray(emb_path_a, dtype=np.float64)
                        if emb_path_a is not None
                        else None,
                        q_vec,
                    )
                )
                if emb_path_a is not None
                else 0.0
            )

            # ── Batch compute f1: source_header vs all target headers ──
            if a_vec is not None:
                f1_all = cosine_sim_batch(target_emb_matrix, a_vec)
                f1_all[~target_emb_valid] = 0.0
            else:
                f1_all = np.zeros(n_target, dtype=np.float64)

            for i_t, h_t in enumerate(target.header_list):
                pre_f1 = float(f1_all[i_t])
                pre_sim_b = float(target_query_sims[i_t])
                pre_sim_path_b = float(target_path_query_sims[i_t])
                pre_q_b_tok_jaccard = (
                    float(target_q_b_tok_jaccard[i_t])
                    if target_q_b_tok_jaccard is not None
                    else None
                )
                pre_q_b_tok_recall = (
                    float(target_q_b_tok_recall[i_t])
                    if target_q_b_tok_recall is not None
                    else None
                )
                pre_q_b_tok_precision = (
                    float(target_q_b_tok_precision[i_t])
                    if target_q_b_tok_precision is not None
                    else None
                )
                pre_q_b_char3_jaccard = (
                    float(target_q_b_char3_jaccard[i_t])
                    if target_q_b_char3_jaccard is not None
                    else None
                )
                pre_q_b_fuzz_ratio = (
                    float(target_q_b_fuzz_ratio[i_t])
                    if target_q_b_fuzz_ratio is not None
                    else None
                )
                pre_q_path_b_tok_jaccard = (
                    float(target_q_path_b_tok_jaccard[i_t])
                )
                pre_bm25_header_b = float(target_bm25_header_scores[i_t])
                pre_bm25_combined_b = float(target_bm25_combined_scores[i_t])

                features = extract_ml_features(
                    header_s_refined,
                    h_t,
                    idx_s,
                    i_t,
                    source.context,
                    target.context,
                    source.embeddings,
                    target.embeddings,
                    query_embedding,
                    query_text=query_text,
                    mode=RETAINED_MODE,
                    _pre_f1=pre_f1,
                    _pre_sim_a=pre_sim_a,
                    _pre_sim_b=pre_sim_b,
                    _pre_sim_path_a=pre_sim_path_a,
                    _pre_sim_path_b=pre_sim_path_b,
                    _pre_bm25_header_b=pre_bm25_header_b,
                    _pre_bm25_combined_b=pre_bm25_combined_b,
                    _pre_q_b_tok_jaccard=pre_q_b_tok_jaccard,
                    _pre_q_b_tok_recall=pre_q_b_tok_recall,
                    _pre_q_b_tok_precision=pre_q_b_tok_precision,
                    _pre_q_b_char3_jaccard=pre_q_b_char3_jaccard,
                    _pre_q_b_fuzz_ratio=pre_q_b_fuzz_ratio,
                    _pre_q_path_b_tok_jaccard=pre_q_path_b_tok_jaccard,
                    bm25_scorer_b=target_bm25,
                )
                feature_rows.append(features)
                feature_target_indices.append(i_t)

    # 3. Predict & Score Aggregation
    # Collect predictions from all sources (supports softmax / top2_mean aggregation)
    use_adaptive = True
    model_scores_collected: Dict[str, Dict[int, list]] = (
        {model_type: {} for model_type, _, _ in models}
    )
    model_scores: Dict[str, Dict[int, float]] = {
        model_type: {} for model_type, _, _ in models
    }

    if not feature_rows:
        raise ValueError(
            "No mode-25 feature rows were generated for this target document."
        )

    # Convert to numpy and predict
    for model_type, _, model in models:
        if not model.feature_names:
            continue
        feature_names = model.feature_names
        extracted_names = set(feature_rows[0].keys())
        missing = set(feature_names) - extracted_names
        if missing:
            print(
                f"  Warning: {model_type} requires features not in extraction: {missing}"
            )
        X_list = [[row.get(name, 0.0) for name in feature_names] for row in feature_rows]
        X = np.array(X_list)
        predictions = model.predict(
            X, query_embedding=query_embedding, target_doc_id=target.doc_id
        )
        for pred, idx in zip(predictions, feature_target_indices):
            _collect_score(model_scores_collected[model_type], idx, pred)

    # Adaptive aggregation: aggregate collected score lists into final scores
    if use_adaptive:
        for mt in model_scores_collected:
            model_scores[mt] = _aggregate_scores(
                model_scores_collected[mt],
                method=score_agg,
                softmax_alpha=softmax_alpha,
            )

    # 4. Rank
    method_results = {}

    # Ground Truth Details (common)
    gt_details = []
    for idx in target.prov_indices:
        h = target.header_list[idx]
        gt_details.append(
            {
                "idx": idx,
                "text": h.text,
                "text_span": h.text_span,
                "path_text": h.processing_path,
            }
        )

    # Helper to build result for a method
    def build_method_result(scores_list: List[Tuple[int, float]]) -> Dict[str, Any]:
        # Fix: empty scores_list is an abnormal condition; raise a clear error
        if not scores_list:
            raise ValueError(
                "scores_list is empty; cannot compute ranking. "
                "Possible causes: (1) model not loaded (2) feature extraction failed (3) no valid source-target pairs"
            )

        # scores_list: [(idx, score), ...] unsorted
        sorted_scores = sorted(scores_list, key=lambda x: x[1], reverse=True)

        # Calculate Rank
        rank = next(
            (
                i + 1
                for i, (idx, _) in enumerate(sorted_scores)
                if idx in target.prov_indices
            ),
            len(sorted_scores) + 1,
        )

        # Get Top 5
        top_5 = []
        for i in range(min(5, len(sorted_scores))):
            idx, score = sorted_scores[i]
            h = target.header_list[idx]
            top_5.append(
                {
                    "idx": idx,
                    "score": float(score),
                    "text": h.text,
                    "text_span": h.text_span,
                    "path_text": h.processing_path,
                }
            )

        return {"rank": rank, "total_candidates": len(scores_list), "top_5": top_5}

    # RAG Results
    method_results["RAG"] = build_method_result(rag_scores)

    # Model Results
    for model_type, scores_dict in model_scores.items():
        # scores_dict is {list_index: score}
        scores_list = list(scores_dict.items())
        if not scores_list:
            # Print warning and skip this model to avoid crashing
            warnings.warn(
                f"[{model_type}] No prediction scores generated for target document {target.doc_id}. "
                f"Check: (1) model exists (2) source_docs have prov_indices (3) feature extraction succeeded"
            )
            continue  # Skip this model; do not add to method_results
        method_results[model_type] = build_method_result(scores_list)

    result = {
        "doc_id": target.doc_id,
        "ground_truth_details": gt_details,
        "methods": method_results,
    }
    if return_raw_scores:
        result["raw_scores"] = {mt: dict(scores) for mt, scores in model_scores.items()}
    return result


def evaluate_question(
    dataset: str,
    q_idx: int,
    model_configs: List[str],
    experiment: str,
    provider: str,
    seeds: List[int] = None,
    progress_queue: Optional[Any] = None,
    max_target_docs: Optional[int] = None,
    xgb_device: str = "auto",
    score_agg: str = "softmax",
    softmax_alpha: float = 5.0,
    nn_device: str = "auto",
    parser: str = "docling",
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)
    xgb_device = _resolve_xgb_device(xgb_device)

    splits_dir = paths.get_splits_dir(dataset)
    processing_dir = paths.get_processing_dir(dataset)
    embeddings_dir = paths.get_embeddings_dir(dataset, provider=provider)

    # Load Labels from Split Dir
    source_labels_path = splits_dir / f"q{q_idx}_train_labels.json"
    target_labels_path = splits_dir / f"q{q_idx}_test_labels.json"

    source_labels = load_labels(source_labels_path)
    target_labels = load_labels(target_labels_path)
    if max_target_docs is not None and max_target_docs > 0:
        target_labels = target_labels[:max_target_docs]

    if not source_labels or not target_labels:
        return {"error": "no_labels", "q_idx": q_idx}

    question = source_labels[0].get("question", "")

    # Load Models
    models = []
    for model_type in model_configs:
        mode = MODEL_TYPE_TO_MODE.get(model_type)
        if mode is None:
            continue

        # Load Ensemble
        loaded_models = []
        for seed in seeds:
            models_root = paths.get_models_dir(dataset)
            model_path = (
                models_root
                / model_type
                / f"{dataset}_{provider}_{paths.reconstructed_tag}"
                / f"q{q_idx}_seed{seed}.json"
            )

            if model_path.exists():
                loaded_model = load_classifier(str(model_path), device=nn_device)
                _apply_xgb_device_override(loaded_model, xgb_device)
                loaded_models.append(loaded_model)

        if loaded_models:
            models.append((model_type, mode, EnsembleModel(loaded_models)))

    # Load Docs
    source_docs = []
    for label in source_labels:
        doc_data = load_doc_data(
            label,
            processing_dir,
            embeddings_dir,
            provider=provider,
        )
        if doc_data:
            source_docs.append(doc_data)

    target_docs = []
    for label in target_labels:
        doc_data = load_doc_data(
            label,
            processing_dir,
            embeddings_dir,
            provider=provider,
        )
        if doc_data:
            target_docs.append(doc_data)

    if not source_docs or not target_docs:
        return {"error": "no_docs", "q_idx": q_idx}

    # Query Embedding
    m_name = get_model_name_for_provider(provider)
    q_emb = get_query_embedding(
        question, model=m_name, provider=provider, source=dataset
    )
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()

    # Evaluate Targets
    from collections import defaultdict

    all_ranks = defaultdict(list)
    target_details = []

    total_targets = len(target_docs)
    if progress_queue:
        progress_queue.put(("progress", q_idx, 0, total_targets))

    for i, target_doc in enumerate(target_docs):
        # Build per-doc model list: global
        doc_models = list(models)
        effective_sources = source_docs

        doc_res = evaluate_target_doc(
            target_doc,
            effective_sources,
            q_emb,
            question,
            doc_models,
            source_labels,
            score_agg=score_agg,
            softmax_alpha=softmax_alpha,
        )
        target_details.append(doc_res)

        # Extract ranks for metrics calculation (including dynamically added RRF methods)
        for method, res in doc_res["methods"].items():
            rank = res["rank"]
            total = res["total_candidates"]
            all_ranks[method].append((rank, total))

        if progress_queue:
            progress_queue.put(("progress", q_idx, i + 1, total_targets))

    metrics = {name: compute_metrics(ranks) for name, ranks in all_ranks.items()}

    return {
        "q_idx": q_idx,
        "question": question,
        "metrics": metrics,
        "num_target_docs": len(target_docs),
        "target_docs_details": target_details,
    }


def _evaluate_worker(args):
    (
        dataset,
        q_idx,
        model_configs,
        experiment,
        provider,
        seeds,
        progress_queue,
        max_target_docs,
        xgb_device,
        score_agg,
        softmax_alpha,
        nn_device,
        parser,
    ) = args
    try:
        t0 = time.perf_counter()
        res = evaluate_question(
            dataset,
            q_idx,
            model_configs,
            experiment,
            provider,
            seeds=seeds,
            progress_queue=progress_queue,
            max_target_docs=max_target_docs,
            xgb_device=xgb_device,
            score_agg=score_agg,
            softmax_alpha=softmax_alpha,
            nn_device=nn_device,
            parser=parser,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res["_eval_elapsed_ms"] = elapsed_ms
        gc.collect()
        return res
    except Exception as e:
        import traceback

        if progress_queue:
            progress_queue.put(("log", q_idx, f"[red]Error: {e}[/red]"))
        return {
            "error": f"worker_exception: {e}\n{traceback.format_exc()}",
            "q_idx": q_idx,
        }


def compute_overall_metrics(results_list: List[Dict]) -> Dict:
    methods = set()
    for r in results_list:
        if "metrics" in r:
            methods.update(r["metrics"].keys())

    overall = {}
    for method in methods:
        _METRIC_KEYS = [
            "mrr",
            "recall_1",
            "recall_5",
            "recall_10",
            "ndcg_5",
            "ndcg_10",
            "mean_rank",
        ]
        method_metrics = {k: [] for k in _METRIC_KEYS}
        count = 0
        mrr_weighted_sum = 0.0
        mrr_weight_total = 0
        for r in results_list:
            if "metrics" not in r or method not in r["metrics"]:
                continue
            m = r["metrics"][method]
            for k in method_metrics:
                if k in m:
                    method_metrics[k].append(m[k])
            num_target_docs = int(r.get("num_target_docs", 0))
            if num_target_docs > 0 and "mrr" in m:
                mrr_weighted_sum += float(m["mrr"]) * num_target_docs
                mrr_weight_total += num_target_docs
            count += 1

        if count > 0:
            summary = {
                k: (sum(v) / len(v) if v else 0.0) for k, v in method_metrics.items()
            }
            # Legacy field: mrr still represents macro MRR (averaged per question)
            summary["mrr_macro"] = summary["mrr"]
            # micro MRR: weighted by target document count (equivalent to cross-question document-level aggregation)
            summary["mrr_micro"] = (
                mrr_weighted_sum / mrr_weight_total if mrr_weight_total > 0 else 0.0
            )
            summary["count"] = count
            summary["doc_count"] = mrr_weight_total
            overall[method] = summary
    return overall


def evaluate_models(
    dataset: str = "pdfs",
    limit: int = 10,
    model_configs: Optional[List[str]] = None,
    experiment: str = "default",
    workers: Optional[int] = None,
    provider: str = "openrouter",
    seeds: Optional[List[int]] = None,
    max_target_docs: Optional[int] = None,
    xgb_device: str = "auto",
    score_agg: str = "softmax",
    softmax_alpha: float = 5.0,
    nn_device: str = "auto",
    parser: str = "docling",
    questions: Optional[List[int]] = None,
):
    # None -> default ML models; [] -> no ML models, used by rag-v1 baseline.
    model_configs = model_configs if model_configs is not None else DEFAULT_MODEL_TYPES
    seeds_list = seeds if seeds else DEFAULT_SEEDS
    xgb_device = _resolve_xgb_device(xgb_device)
    mp_start_method = _resolve_mp_start_method(xgb_device, nn_device)
    variant = parser if parser != "docling" else None

    paths = PathManager(experiment=experiment, processing_variant=variant)
    results_dir = paths.get_results_dir(dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_str = "+".join(model_configs) if model_configs else "default"

    # Create run directory
    run_name = f"results_{dataset}_{provider}_{paths.reconstructed_tag}_{model_str}_{timestamp}"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    agg_path = run_dir / "results.json"

    q_indices = questions if questions is not None else list(range(limit))
    if not q_indices:
        raise ValueError("No questions selected for evaluation")

    if workers is None:
        workers = min(os.cpu_count() or 1, len(q_indices))

    print("=== Evaluate Models [Problem 1 Step 7] ===")
    print(f"Dataset:    {dataset}")
    print(f"Experiment: {experiment}")
    print(f"Models:     {model_configs}")
    print(f"Seeds:      {seeds_list}")
    print(f"XGB Device: {xgb_device}")
    print(f"NN Device:  {nn_device}")
    print(f"MP Start:   {mp_start_method or 'default'}")
    if max_target_docs is not None and max_target_docs > 0:
        print(f"Max Docs:   {max_target_docs} per question")
    print(f"Results:    {agg_path}")
    print()

    # Collect required embedding keys and sync temp cache
    global _TEMP_CACHE_DIR

    splits_dir = paths.get_splits_dir(dataset)
    processing_dir = paths.get_processing_dir(dataset)
    embeddings_dir = paths.get_embeddings_dir(dataset, provider=provider)
    m_name = get_model_name_for_provider(provider)

    # Temp cache directory
    temp_cache_dir = (
        PROJECT_ROOT
        / ".temp"
        / "embeddings_cache"
        / dataset
        / provider
        / paths.reconstructed_tag
    )
    _TEMP_CACHE_DIR = temp_cache_dir

    print("Collecting needed embedding keys...")
    needed_keys = _collect_needed_keys(
        limit, splits_dir, processing_dir, max_target_docs, q_indices=q_indices
    )
    total_keys = sum(len(keys) for keys in needed_keys.values())
    print(f"Found {len(needed_keys)} docs, {total_keys} embedding keys needed.")

    print("Syncing temp embeddings cache...")
    docs_updated, keys_added, keys_removed = _sync_temp_cache(
        needed_keys,
        embeddings_dir,
        processing_dir,
        temp_cache_dir,
        provider,
    )
    print(
        f"Cache synced: {docs_updated} docs updated, +{keys_added} keys, -{keys_removed} keys.\n"
    )

    # Pre-compute Query Embeddings
    print("Pre-computing query embeddings...")
    for q_idx in q_indices:
        path = splits_dir / f"q{q_idx}_test_labels.json"
        labels = load_labels(path)
        if labels:
            q = labels[0].get("question", "")
            if q:
                get_query_embedding(q, model=m_name, provider=provider, source=dataset)
    print("Done.\n")

    results_list = []

    def _build_eval_tasks(progress_queue):
        return [
            (
                dataset,
                q_idx,
                model_configs,
                experiment,
                provider,
                seeds_list,
                progress_queue,
                max_target_docs,
                xgb_device,
                score_agg,
                softmax_alpha,
                nn_device,
                parser,
            )
            for q_idx in q_indices
        ]

    def _collect_result(_tid, res):
        results_list.append(res)

    run_pool_with_progress(
        build_tasks=_build_eval_tasks,
        worker_fn=_evaluate_worker,
        workers=workers,
        initializer=_worker_init,
        initargs=(str(temp_cache_dir),),
        description="Total Progress",
        result_collector=_collect_result,
        start_method=mp_start_method,
    )

    # Compute Summary
    summary = compute_overall_metrics(results_list)

    # Save individual detail files
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    for res in results_list:
        q_idx = res.get("q_idx", "unknown")
        detail_path = details_dir / f"q{q_idx}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    meta = {
        "dataset": dataset,
        "experiment": experiment,
        "models": model_configs,
        "timestamp": timestamp,
        "provider": provider,
        "xgb_device": xgb_device,
    }

    output = {
        "summary": summary,
        "details": results_list,
        "meta": meta,
    }

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summarize evaluation results
    error_count = sum(1 for r in results_list if "error" in r)
    success_count = len(results_list) - error_count

    print("\n=== Evaluation Complete ===")
    print(
        f"Questions:  {len(results_list)} (Success: {success_count}, Errors: {error_count})"
    )
    print(f"Results:    {agg_path}")

    # Per-question output (grouped by method)
    # Collect all methods
    all_methods = set()
    for r in results_list:
        if "metrics" in r:
            all_methods.update(r["metrics"].keys())

    # Sort results by q_idx
    sorted_results = sorted(
        [r for r in results_list if "metrics" in r], key=lambda x: x.get("q_idx", 0)
    )

    for method in sorted(all_methods):
        print(f"\n--- {method} ---")
        print(
            f"| {'Q':<4} | {'MRR':<8} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'MeanR':<8} | {'#Docs':<6} |"
        )
        print(
            f"|{'-' * 6}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 8}|"
        )
        for r in sorted_results:
            q_idx = r.get("q_idx", "?")
            m = r["metrics"].get(method, {})
            mrr = m.get("mrr", 0)
            r1 = m.get("recall_1", 0)
            r5 = m.get("recall_5", 0)
            r10 = m.get("recall_10", 0)
            ndcg5 = m.get("ndcg_5", 0)
            ndcg10 = m.get("ndcg_10", 0)
            mean_r = m.get("mean_rank", 0)
            n_docs = r.get("num_target_docs", 0)
            print(
                f"| q{q_idx:<3} | {mrr:<8.4f} | {r1:<8.4f} | {r5:<8.4f} | {r10:<8.4f} | {ndcg5:<8.4f} | {ndcg10:<8.4f} | {mean_r:<8.2f} | {n_docs:<6} |"
            )

    print("\n=== Overall Summary ===")
    print(
        f"| {'Method':<25} | {'MRRμ':<8} | {'MRRmacro':<8} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'NDCG@5':<8} | {'NDCG@10':<8} |"
    )
    print(
        f"|{'-' * 27}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|"
    )
    summary_sorted_items = sorted(
        summary.items(),
        key=lambda item: (-item[1].get("mrr_micro", item[1]["mrr"]), item[0]),
    )
    for method, m in summary_sorted_items:
        print(
            f"| {method:<25} | {m.get('mrr_micro', m['mrr']):.4f}   | {m.get('mrr_macro', m['mrr']):.4f}   | {m['recall_1']:.4f}   | {m['recall_5']:.4f}   | {m.get('recall_10', 0):.4f}   | {m.get('ndcg_5', 0):.4f}   | {m.get('ndcg_10', 0):.4f}   |"
        )

    return agg_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, default=10, help="Num questions")
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question indices (0-based), e.g. 0,1,2",
    )
    parser.add_argument(
        "--max-target-docs",
        type=int,
        default=None,
        help="Max target docs per question (truncated by label file order)",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        nargs="+",
        choices=ML_MODEL_TYPES,
        help="Model type(s) (multiple allowed, space-separated)",
    )
    parser.add_argument(
        "--seeds", type=str, help="Comma-separated seed list (e.g., 41,42,43)"
    )
    parser.add_argument(
        "--experiment", type=str, default="default", help="Experiment ID"
    )
    parser.add_argument("--workers", type=int, help="Workers")
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="openrouter",
        choices=EMBED_PROVIDERS,
        help=f"Embedding provider (choices: {', '.join(EMBED_PROVIDERS)})",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost evaluation device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--nn-device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch neural network device (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--score-agg",
        type=str,
        choices=SCORE_AGG_METHODS,
        default="softmax",
        help="Score aggregation: softmax (default, smooth max), top2_mean",
    )
    parser.add_argument(
        "--softmax-alpha",
        type=float,
        default=5.0,
        help="Softmax aggregation temperature alpha (default 5.0, higher = closer to max)",
    )
    parser.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        dest="struct_parser",
        help="Structural parser (docling/mineru)",
    )

    args = parser.parse_args()

    # Parse seeds argument
    seeds_list = None
    if args.seeds:
        seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    questions_list = None
    if args.questions:
        questions_list = [int(q.strip()) for q in args.questions.split(",")]

    evaluate_models(
        dataset=args.dataset,
        limit=args.limit,
        model_configs=args.model_config,
        experiment=args.experiment,
        workers=args.workers,
        provider=args.embed_provider,
        seeds=seeds_list,
        max_target_docs=args.max_target_docs,
        xgb_device=args.xgb_device,
        score_agg=args.score_agg,
        softmax_alpha=args.softmax_alpha,
        nn_device=args.nn_device,
        parser=args.struct_parser,
        questions=questions_list,
    )


if __name__ == "__main__":
    main()
