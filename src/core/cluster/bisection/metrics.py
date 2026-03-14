"""core.cluster.bisection.metrics — Clustering evaluation metrics and utility functions"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def hungarian_map(
    cluster_labels: np.ndarray,
    gt_labels: list[str],
    type_order: list[str],
) -> list[str]:
    """Map cluster IDs to optimal types using Hungarian algorithm.
    
    Resolves the label permutation problem in unsupervised clustering by 
    finding the maximum weight matching between predicted clusters and 
    ground truth classes.
    """
    n_clusters = len(set(cluster_labels))
    n_types = len(type_order)
    type_to_idx = {t: i for i, t in enumerate(type_order)}

    # Contingency matrix
    C = np.zeros((n_clusters, n_types), dtype=int)
    for cl, gt in zip(cluster_labels, gt_labels):
        if gt in type_to_idx:
            C[cl, type_to_idx[gt]] += 1

    # Hungarian (minimize cost using negated values)
    cost = -C.astype(float)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = type_order[c]

    # Fill unmapped clusters with majority vote
    for cl in range(n_clusters):
        if cl not in mapping:
            best = int(np.argmax(C[cl]))
            mapping[cl] = type_order[best]

    return [mapping[cl] for cl in cluster_labels]


def hungarian_map_from_leaves(
    leaf_labels: list[str],
    gt_labels: list[str],
    type_order: list[str],
) -> list[str]:
    """Map leaf cluster string labels to types."""
    unique_leaves = sorted(set(leaf_labels))
    leaf_to_idx = {le: i for i, le in enumerate(unique_leaves)}
    numeric = np.array([leaf_to_idx[le] for le in leaf_labels])
    return hungarian_map(numeric, gt_labels, type_order)


def class_recall(gt: list[str], pred: list[str], cls: str) -> float:
    """Compute single-class recall."""
    tp = sum(1 for g, p in zip(gt, pred) if g == cls and p == cls)
    fn = sum(1 for g, p in zip(gt, pred) if g == cls and p != cls)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def class_precision(gt: list[str], pred: list[str], cls: str) -> float:
    """Compute single-class precision."""
    tp = sum(1 for g, p in zip(gt, pred) if g == cls and p == cls)
    fp = sum(1 for g, p in zip(gt, pred) if g != cls and p == cls)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def merge_periodic(labels: list[str]) -> list[str]:
    """Remap 10K/10Q to PERIODIC.
    
    Groups domain-specific periodic filings into a single category for 
    high-level evaluation of temporal reporting patterns.
    """
    return ["PERIODIC" if t in ("10K", "10Q") else t for t in labels]


def save_confusion_csv(
    gt: list[str],
    pred: list[str],
    labels: list[str],
    path: Path,
) -> None:
    """Save confusion matrix as CSV."""
    cm = confusion_matrix(gt, pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(path)


def save_assignments(
    result: dict,
    data: dict,
    out_dir: Path,
    filename: str = "recursive_assignments.csv",
) -> None:
    """Save clustering assignments CSV."""
    all_doc_ids = data["all_doc_ids"]
    gt_list = result.get("gt", [])
    pred_list = result.get("pred", [])

    rows = []
    for d, gt, pred in zip(all_doc_ids, gt_list, pred_list):
        rows.append(
            {
                "doc_id": d,
                "predicted_type": pred,
                "gt_type": gt,
                "correct": 1 if pred == gt else 0,
            }
        )

    with open(out_dir / filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_id", "predicted_type", "gt_type", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrices(
    flat_results: dict,
    recursive_results: dict,
    data: dict,
    fig_dir: Path,
    semisup_recursive_result: dict | None = None,
) -> None:
    """Plot confusion matrix comparison chart (full corpus 603 docs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.cluster.common.paths import TYPE_ORDER_FULL

    methods = {
        "Flat k=5 (Full)": (
            flat_results["flat_full_k5"]["gt"],
            flat_results["flat_full_k5"]["pred"],
            TYPE_ORDER_FULL,
        ),
        "Recursive (Full)": (
            recursive_results["recursive_full_pruned"]["gt"],
            recursive_results["recursive_full_pruned"]["pred"],
            TYPE_ORDER_FULL,
        ),
    }
    if semisup_recursive_result is not None:
        methods["Recursive+SemiSup"] = (
            semisup_recursive_result["gt"],
            semisup_recursive_result["pred"],
            TYPE_ORDER_FULL,
        )

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods + 2, 5))
    if n_methods == 1:
        axes = [axes]
    for ax, (name, (gt, pred, labels)) in zip(axes, methods.items(), strict=False):
        cm = confusion_matrix(gt, pred, labels=labels)
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices_comparison.png", dpi=300)
    plt.savefig(fig_dir / "confusion_matrices_comparison.pdf")
    plt.close()
    logger.info("  Saved confusion matrix figure (full corpus)")


__all__ = [
    "hungarian_map",
    "hungarian_map_from_leaves",
    "class_recall",
    "class_precision",
    "merge_periodic",
    "save_confusion_csv",
    "save_assignments",
    "plot_confusion_matrices",
]
