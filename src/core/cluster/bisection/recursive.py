"""core.cluster.bisection.recursive — Recursive spectral bisection core functions"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from core.cluster.common.paths import BISECTION_OUT_DIR, SEED

logger = logging.getLogger(__name__)

FIG_DIR = BISECTION_OUT_DIR / "figures"


def recursive_bisect_pruned(
    S: np.ndarray,
    doc_ids: np.ndarray,
    min_cluster_size: int = 15,
    sil_threshold: float = 0.06,
    max_depth: int = 6,
    gt_labels: dict[str, str] | None = None,
    fig_path: Path | None = None,
) -> tuple[dict[str, str], int, list[dict]]:
    """Recursive spectral bisection + silhouette pruning (SOTA core method).

    Implements a top-down hierarchical clustering approach where each cluster 
    is recursively split into two using spectral clustering. Splits are 
    pruned (stopped) if the resulting silhouette score falls below a threshold, 
    preventing over-segmentation of the document corpus.
    """
    global_idx = {d: i for i, d in enumerate(doc_ids)}
    histories: list[dict] = []

    def _get_sub_S(sub_ids: np.ndarray) -> np.ndarray:
        idx = [global_idx[d] for d in sub_ids]
        return S[np.ix_(idx, idx)]

    def _recurse(
        sub_ids: np.ndarray,
        depth: int,
        label: str,
    ) -> dict[str, str]:
        """Core recursion: split current subset or return as leaf node."""
        n = len(sub_ids)
        indent = "  " * (depth + 2)

        # Termination: too small or too deep
        if n < min_cluster_size * 2 or depth >= max_depth:
            if depth > 0:
                reason = "max_depth" if depth >= max_depth else f"too_small({n})"
                logger.info(f"{indent}d{depth} {label}: {n} docs → leaf ({reason})")
            return {d: label for d in sub_ids}

        S_sub = _get_sub_S(sub_ids)
        D_sub = np.clip(1.0 - S_sub, 0.0, None)
        np.fill_diagonal(D_sub, 0.0)

        try:
            sc = SpectralClustering(
                n_clusters=2,
                affinity="precomputed",
                random_state=SEED,
            )
            labels = sc.fit_predict(S_sub)
            if len(set(labels)) < 2:
                return {d: label for d in sub_ids}
            sil = silhouette_score(D_sub, labels, metric="precomputed")
        except Exception:
            return {d: label for d in sub_ids}

        # Silhouette gating
        if sil < sil_threshold:
            logger.info(
                f"{indent}d{depth} {label}: {n} docs, "
                f"sil={sil:.4f} < {sil_threshold} → leaf"
            )
            return {d: label for d in sub_ids}

        # GT logging (optional)
        gt_info = ""
        if gt_labels is not None:
            gt_here = [gt_labels.get(d, "?") for d in sub_ids]
            split_nmi = normalized_mutual_info_score(gt_here, labels)
            gt_info = f", NMI_gt={split_nmi:.3f}"

        histories.append(
            {
                "depth": depth,
                "node": label,
                "n_docs": n,
                "split_sil": round(sil, 4),
            }
        )
        logger.info(
            f"{indent}d{depth} {label}: {n} docs → bisect (sil={sil:.4f}{gt_info})"
        )

        result: dict[str, str] = {}
        for i in range(2):
            child_mask = labels == i
            child_ids = sub_ids[child_mask]
            if len(child_ids) == 0:
                continue
            result.update(_recurse(child_ids, depth + 1, f"{label}_{i}"))
        return result

    logger.info("    Recursive bisect + pruning starting...")
    raw = _recurse(doc_ids, 0, "H")

    # Relabel to consecutive labels
    unique = sorted(set(raw.values()))
    k = len(unique)
    relabel = {old: f"H_{i}" for i, old in enumerate(unique)}
    final = {d: relabel[lbl] for d, lbl in raw.items()}
    logger.info(f"    Recursive bisect + pruning done: {k} clusters")

    # Save split tree figure
    if fig_path and histories:
        fig, ax = plt.subplots(figsize=(10, 5))
        nodes = sorted(histories, key=lambda h: (h["depth"], h["node"]))
        labels_plot = [f"d{h['depth']}:{h['node']}\n{h['n_docs']}d" for h in nodes]
        sils = [h["split_sil"] for h in nodes]
        colors = ["#2196F3" if s >= sil_threshold else "#FF5722" for s in sils]
        x = range(len(nodes))
        ax.bar(x, sils, color=colors, alpha=0.8)
        ax.axhline(
            sil_threshold,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"threshold={sil_threshold}",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels_plot, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Silhouette @ k=2")
        ax.set_title(f"Recursive Bisection + Pruning: {k} final clusters")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"    Figure saved → {fig_path}")

    return final, k, histories


__all__ = ["recursive_bisect_pruned"]
