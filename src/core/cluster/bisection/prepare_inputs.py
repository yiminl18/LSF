"""core.cluster.bisection.prepare_inputs -- Build canonical Problem 2 inputs."""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from core.cluster.common.io import extract_headers, get_doc_ids, load_json
from core.cluster.common.paths import (
    BISECTION_LABELS_PATH,
    BISECTION_REPR_PATH,
    BISECTION_SIM_PATH,
    CHI_PROCESSING,
    D_EMB,
    DEFAULT_EMBED_PROVIDER,
    FILING_CSV,
    PHASE1_REPR_DIR,
    SEC_PROCESSING,
    SEED,
    get_embedding_dir,
)
from core.cluster.common.io import classify_doc_type
from core.cluster.bisection.similarity import build_semantic_similarity_matrix

logger = logging.getLogger(__name__)


def _load_filing_types(path: Path) -> dict[str, str]:
    filing_types: dict[str, str] = {}
    if not path.exists():
        return filing_types
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filing_types[row["doc_id"]] = row["filing_type"]
    return filing_types


def _load_node_embeddings(
    doc_id: str,
    sample_types: dict[str, str],
    embed_provider: str,
) -> dict[str, np.ndarray]:
    emb_dir = get_embedding_dir(doc_id, sample_types, provider=embed_provider)
    emb_path = emb_dir / f"{doc_id}_reconstructed_embeddings.npz"
    if not emb_path.exists():
        return {}
    data = np.load(emb_path, allow_pickle=True)
    return {str(key): data["values"][idx] for idx, key in enumerate(data["keys"])}


def _match_header_embedding(
    header: dict[str, Any],
    emb_dict: dict[str, np.ndarray],
) -> np.ndarray | None:
    text = (header.get("text") or "").strip()
    span = (header.get("text_span") or "").strip()
    path_text = ((header.get("structure") or {}).get("path_text") or "").strip()

    candidates = []
    if path_text and span:
        candidates.append(f"{path_text} {span}")
    if text and span:
        candidates.append(f"{text} {span}")
    if path_text:
        candidates.append(path_text)
    if text:
        candidates.append(text)

    for candidate in candidates:
        embedding = emb_dict.get(candidate)
        if embedding is not None and len(embedding) == D_EMB:
            return np.asarray(embedding, dtype=np.float32)
    return None


def build_sec_filing_types_csv(force: bool = False) -> dict[str, str]:
    """Write output/phase0/sec_filing_types.csv for the canonical SEC corpus."""
    if FILING_CSV.exists() and not force:
        return _load_filing_types(FILING_CSV)

    sec_doc_ids = get_doc_ids(SEC_PROCESSING)
    filing_types = {doc_id: classify_doc_type(doc_id) for doc_id in sec_doc_ids}

    FILING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FILING_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "filing_type", "extraction_method"])
        for doc_id in sorted(filing_types):
            writer.writerow([doc_id, filing_types[doc_id], "filename_regex"])

    logger.info("Wrote %s filing types to %s", len(filing_types), FILING_CSV)
    return filing_types


def build_full_representations(
    force: bool = False,
    reuse_phase1_cache: bool = True,
    embed_provider: str = DEFAULT_EMBED_PROVIDER,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    """Build full-corpus representations used by canonical Problem 2.
    
    Extracts and caches the structural nodes and semantic embeddings for every 
    document in the corpus to serve as the foundation for similarity matrices.
    """
    filing_types = build_sec_filing_types_csv(force=force)
    sec_doc_ids = get_doc_ids(SEC_PROCESSING)
    chi_doc_ids = get_doc_ids(CHI_PROCESSING)
    all_doc_ids = sorted(sec_doc_ids) + sorted(chi_doc_ids)
    sample_types = {doc_id: filing_types[doc_id] for doc_id in sec_doc_ids}
    sample_types.update({doc_id: "CHI" for doc_id in chi_doc_ids})

    if BISECTION_REPR_PATH.exists() and not force:
        with open(BISECTION_REPR_PATH, "rb") as handle:
            representations = pickle.load(handle)
        valid_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id in representations]
        return representations, valid_doc_ids, sample_types

    representations: dict[str, dict[str, Any]] = {}
    phase1_cache = PHASE1_REPR_DIR / "all_representations.pkl"
    if reuse_phase1_cache and phase1_cache.exists():
        with open(phase1_cache, "rb") as handle:
            representations.update(pickle.load(handle))

    missing_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in representations]
    for doc_id in tqdm(missing_doc_ids, desc="P2 representations"):
        proc_dir = CHI_PROCESSING if sample_types.get(doc_id) == "CHI" else SEC_PROCESSING
        data = load_json(proc_dir / f"{doc_id}_reconstructed.json")
        headers = extract_headers(data.get("texts", []))
        if not headers:
            continue

        emb_dict = _load_node_embeddings(doc_id, sample_types, embed_provider)
        semantic = np.zeros((len(headers), D_EMB), dtype=np.float32)
        emb_mask = np.zeros(len(headers), dtype=bool)
        nodes = []
        for idx, (_, header) in enumerate(headers):
            embedding = _match_header_embedding(header, emb_dict)
            if embedding is not None:
                semantic[idx] = embedding
                emb_mask[idx] = True
            nodes.append(
                {
                    "text": (header.get("text") or "").strip(),
                    "depth": int((header.get("structure") or {}).get("depth", 0)),
                }
            )

        representations[doc_id] = {
            "semantic": semantic,
            "emb_mask": emb_mask,
            "nodes": nodes,
            "n_nodes": len(nodes),
        }

    BISECTION_REPR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BISECTION_REPR_PATH, "wb") as handle:
        pickle.dump(representations, handle)

    valid_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id in representations]
    logger.info("Wrote %s representations to %s", len(valid_doc_ids), BISECTION_REPR_PATH)
    return representations, valid_doc_ids, sample_types


def build_full_corpus_labels(
    valid_doc_ids: list[str],
    sample_types: dict[str, str],
    S_full: np.ndarray,
    force: bool = False,
) -> np.ndarray:
    """Cluster the full corpus and write output/phase2/clustering/full_corpus_labels.npz."""
    from sklearn.cluster import SpectralClustering
    if BISECTION_LABELS_PATH.exists() and not force:
        data = np.load(BISECTION_LABELS_PATH, allow_pickle=True)
        return data["labels"]

    sec_doc_ids = [doc_id for doc_id in valid_doc_ids if sample_types.get(doc_id) != "CHI"]
    n_clusters = len({sample_types[doc_id] for doc_id in sec_doc_ids})
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=SEED,
    )
    labels = clusterer.fit_predict(S_full)

    BISECTION_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        BISECTION_LABELS_PATH,
        labels=labels,
        doc_ids=np.asarray(valid_doc_ids, dtype=object),
    )
    logger.info("Wrote %s corpus labels to %s", len(labels), BISECTION_LABELS_PATH)
    return labels


def prepare_canonical_inputs(
    force: bool = False,
    reuse_phase1_cache: bool = True,
    embed_provider: str = DEFAULT_EMBED_PROVIDER,
) -> dict[str, Any]:
    """Generate the four canonical Problem 2 runtime artifacts."""
    start = time.time()
    representations, valid_doc_ids, sample_types = build_full_representations(
        force=force,
        reuse_phase1_cache=reuse_phase1_cache,
        embed_provider=embed_provider,
    )
    similarity = build_semantic_similarity_matrix(
        representations,
        valid_doc_ids,
        force=force,
    )
    labels = build_full_corpus_labels(
        valid_doc_ids,
        sample_types,
        similarity,
        force=force,
    )
    summary = {
        "n_all": len(valid_doc_ids),
        "n_sec": sum(1 for doc_id in valid_doc_ids if sample_types.get(doc_id) != "CHI"),
        "n_chi": sum(1 for doc_id in valid_doc_ids if sample_types.get(doc_id) == "CHI"),
        "label_count": len(labels),
        "similarity_shape": tuple(similarity.shape),
        "elapsed_s": round(time.time() - start, 2),
        "embed_provider": embed_provider,
        "filing_csv": str(FILING_CSV),
        "representations": str(BISECTION_REPR_PATH),
        "semantic_matrix": str(BISECTION_SIM_PATH),
        "labels_npz": str(BISECTION_LABELS_PATH),
    }
    logger.info("Canonical P2 inputs ready: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical Problem 2 inputs")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if they already exist")
    parser.add_argument(
        "--no-reuse-phase1-cache",
        action="store_true",
        help="Do not warm-start from output/phase1/representations/all_representations.pkl",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default=DEFAULT_EMBED_PROVIDER,
        help="Embedding cache provider used to read document embeddings (default: openrouter)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = prepare_canonical_inputs(
        force=args.force,
        reuse_phase1_cache=not args.no_reuse_phase1_cache,
        embed_provider=args.embed_provider,
    )
    print(summary)


__all__ = [
    "build_sec_filing_types_csv",
    "build_full_representations",
    "build_full_corpus_labels",
    "prepare_canonical_inputs",
    "main",
]
