# -*- coding: utf-8 -*-
"""
ML evaluation metrics.

Provides ranking metrics such as MRR for training early stopping and
model evaluation.
"""

from collections import defaultdict
from typing import List

import numpy as np


def compute_mrr_for_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    doc_ids: List[str],
) -> float:
    """
    Compute MRR (Mean Reciprocal Rank) on the validation set.

    Groups samples by source document and computes the mean reciprocal
    rank of positive samples within each group.

    Args:
        y_true: Ground-truth labels (0/1)
        y_pred: Predicted probabilities
        doc_ids: Document ID per sample (format: "{source_doc}_{target_doc}")

    Returns:
        MRR value [0, 1]
    """
    # Group by source document
    source_groups = defaultdict(list)
    for doc_id, label, pred in zip(doc_ids, y_true, y_pred):
        # doc_id format: "doc1_doc2"; use doc1 as source
        source = doc_id.rsplit("_", 1)[0]
        source_groups[source].append((label, pred))

    # Compute rank of positive samples in each group
    ranks = []
    for samples in source_groups.values():
        # Sort by predicted probability descending
        sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)
        for rank, (label, _) in enumerate(sorted_samples, 1):
            if label == 1:
                ranks.append(rank)

    if not ranks:
        return 0.0

    # MRR = mean reciprocal rank
    return float(np.mean([1.0 / r for r in ranks]))
