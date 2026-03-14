"""core.cluster.bisection.pipeline -- Canonical fused clustering pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from typing import Any

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from core.cluster.bisection.data import load_all_data
from core.cluster.bisection.similarity import (
    build_heading_tfidf_similarity,
    build_tree_shape_similarity,
)
from core.cluster.bisection.llm_merge import llm_merge_clusters
from core.cluster.bisection.metrics import (
    class_precision,
    class_recall,
    hungarian_map_from_leaves,
    merge_periodic,
    save_assignments,
    save_confusion_csv,
)
from core.cluster.bisection.recursive import recursive_bisect_pruned
from core.cluster.common.paths import (
    BISECTION_OUT_DIR,
    TYPE_ORDER_FULL,
    TYPE_ORDER_MERGED,
)
from core.llm.model import LLM_PROVIDERS

logger = logging.getLogger(__name__)

FUSION_WEIGHTS = {
    "sem": 0.5,
    "tfidf": 0.3,
    "tree": 0.2,
}


def _evaluate_assignments(
    label: str,
    assignments: dict[str, str],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a clustering assignment against the retained metrics.
    
    Computes both standard clustering metrics (NMI, ARI) and domain-specific 
    recall/precision for periodic document types.
    """
    all_doc_ids = data["all_doc_ids"]
    gt_full = [data["gt_4class"][doc_id] for doc_id in all_doc_ids]
    raw_pred = [assignments[doc_id] for doc_id in all_doc_ids]
    mapped_pred = hungarian_map_from_leaves(raw_pred, gt_full, TYPE_ORDER_FULL)

    periodic_gt = merge_periodic(gt_full)
    periodic_pred = merge_periodic(mapped_pred)

    return {
        "label": label,
        "k": len(set(assignments.values())),
        "NMI": normalized_mutual_info_score(gt_full, raw_pred),
        "ARI": adjusted_rand_score(gt_full, raw_pred),
        "10Q_recall": class_recall(gt_full, mapped_pred, "10Q"),
        "PERIODIC_recall": class_recall(periodic_gt, periodic_pred, "PERIODIC"),
        "PERIODIC_precision": class_precision(periodic_gt, periodic_pred, "PERIODIC"),
        "gt": gt_full,
        "pred": mapped_pred,
        "assignments": assignments,
    }


def _save_outputs(results: dict[str, dict[str, Any]], data: dict[str, Any]) -> None:
    """Persist the canonical artifact outputs for the fused pipeline."""
    BISECTION_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "weights": FUSION_WEIGHTS,
        "results": {},
    }
    for name, result in results.items():
        rows.append(
            {
                "method": name,
                "k": result["k"],
                "NMI": round(result["NMI"], 4),
                "ARI": round(result["ARI"], 4),
                "10Q_recall": round(result["10Q_recall"], 4),
                "PERIODIC_recall": round(result["PERIODIC_recall"], 4),
                "PERIODIC_precision": round(result["PERIODIC_precision"], 4),
            }
        )
        summary["results"][name] = rows[-1]

        save_confusion_csv(
            result["gt"],
            result["pred"],
            TYPE_ORDER_FULL,
            BISECTION_OUT_DIR / f"{name}_confusion.csv",
        )
        save_confusion_csv(
            merge_periodic(result["gt"]),
            merge_periodic(result["pred"]),
            TYPE_ORDER_MERGED,
            BISECTION_OUT_DIR / f"{name}_periodic_confusion.csv",
        )
        save_assignments(
            result,
            data,
            BISECTION_OUT_DIR,
            filename=f"{name}_assignments.csv",
        )

    with open(BISECTION_OUT_DIR / "canonical_pipeline_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(BISECTION_OUT_DIR / "canonical_pipeline_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def run_recursive_fused(
    data: dict[str, Any],
    llm_provider: str = "azure",
) -> dict[str, Any]:
    """Run the retained P2 workflow with fixed fusion weights and one LLM merge.
    
    Orchestrates the full Problem 2 pipeline: matrix fusion (semantic + text 
    + tree), recursive spectral bisection, and semantic LLM refinement.
    """
    logger.info("=" * 60)
    logger.info("Problem 2: canonical fused clustering pipeline")

    semantic_similarity = data["S_full"]
    all_doc_ids = data["all_doc_ids"]
    representations = data["representations"]

    logger.info("  Building heading TF-IDF similarity matrix...")
    tfidf_similarity, _ = build_heading_tfidf_similarity(all_doc_ids, representations)

    logger.info("  Building tree-shape similarity matrix...")
    tree_similarity = build_tree_shape_similarity(all_doc_ids, representations)

    fused_similarity = (
        FUSION_WEIGHTS["sem"] * semantic_similarity
        + FUSION_WEIGHTS["tfidf"] * tfidf_similarity
        + FUSION_WEIGHTS["tree"] * tree_similarity
    )

    logger.info(
        "  Running recursive bisection with fixed weights sem=%.1f tfidf=%.1f tree=%.1f",
        FUSION_WEIGHTS["sem"],
        FUSION_WEIGHTS["tfidf"],
        FUSION_WEIGHTS["tree"],
    )
    pruned_assignments, _, _ = recursive_bisect_pruned(
        fused_similarity,
        all_doc_ids,
        gt_labels=data["gt_4class"],
    )
    pruned_result = _evaluate_assignments("fused_pruned", pruned_assignments, data)

    logger.info(
        "  Fused pruning: k=%s NMI=%.4f ARI=%.4f 10Q=%.1f%%",
        pruned_result["k"],
        pruned_result["NMI"],
        pruned_result["ARI"],
        pruned_result["10Q_recall"] * 100.0,
    )

    logger.info("  Running single corpus-level LLM merge with provider=%s...", llm_provider)
    merged_assignments = llm_merge_clusters(
        pruned_assignments,
        representations,
        llm_provider=llm_provider,
    )
    merged_result = _evaluate_assignments("llm_merged", merged_assignments, data)

    logger.info(
        "  LLM merged: k=%s NMI=%.4f ARI=%.4f 10Q=%.1f%%",
        merged_result["k"],
        merged_result["NMI"],
        merged_result["ARI"],
        merged_result["10Q_recall"] * 100.0,
    )

    results = {
        "fused_pruned": pruned_result,
        "llm_merged": merged_result,
    }
    _save_outputs(results, data)

    return {
        "weights": dict(FUSION_WEIGHTS),
        "results": results,
    }


def main() -> None:
    """Run the retained artifact-facing P2 workflow."""
    parser = argparse.ArgumentParser(description="Run the canonical Problem 2 pipeline")
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=sorted(LLM_PROVIDERS),
        default="azure",
        help="LLM provider for the single corpus-level merge step",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = load_all_data()
    run_recursive_fused(data, llm_provider=args.llm_provider)
    logger.info("Canonical P2 outputs saved to %s", BISECTION_OUT_DIR)


__all__ = [
    "FUSION_WEIGHTS",
    "run_recursive_fused",
    "main",
]
