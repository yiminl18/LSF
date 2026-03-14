"""core.cluster.bisection.data -- Data loading."""

from __future__ import annotations

import csv
import logging
import pickle
from collections import Counter
from typing import Any

import numpy as np

from core.cluster.common.paths import (
    BISECTION_LABELS_PATH,
    BISECTION_REPR_PATH,
    BISECTION_SIM_PATH,
    FILING_CSV,
)
from core.cluster.common.io import classify_doc_type

logger = logging.getLogger(__name__)


def load_all_data() -> dict[str, Any]:
    """Load all precomputed data and build multi-class ground truth.
    
    Combines the semantic similarity matrix, clustering labels, and document 
    representations, while inferring filing types to create evaluation labels.
    """
    logger.info("Loading data...")

    S_full = np.load(BISECTION_SIM_PATH)
    data = np.load(BISECTION_LABELS_PATH, allow_pickle=True)
    all_doc_ids = np.array([str(d) for d in data["doc_ids"]])

    with open(BISECTION_REPR_PATH, "rb") as f:
        representations = pickle.load(f)

    # Load SEC filing types
    filing_types: dict[str, str] = {}
    with open(FILING_CSV) as f:
        for row in csv.DictReader(f):
            filing_types[row["doc_id"]] = row["filing_type"]

    # Build ground truth: 5-class and 4-class
    gt_5class: dict[str, str] = {}
    gt_4class: dict[str, str] = {}  # ANNUAL → 10K
    for d in all_doc_ids:
        if d in filing_types:
            t = filing_types[d]
        else:
            t = classify_doc_type(d)
            if t == "OTHER":
                t = "CHI"
        gt_5class[d] = t
        gt_4class[d] = "10K" if t == "ANNUAL" else t

    # Indices
    sec_mask = np.array([gt_5class[d] != "CHI" for d in all_doc_ids])
    chi_mask = ~sec_mask
    sec_indices = np.where(sec_mask)[0]
    sec_doc_ids = all_doc_ids[sec_mask]
    S_sec = S_full[np.ix_(sec_indices, sec_indices)]

    type_counts = Counter(gt_4class[d] for d in sec_doc_ids)
    logger.info(f"SEC docs: {len(sec_doc_ids)}, 4-class: {dict(type_counts)}")
    logger.info(f"CHI docs: {chi_mask.sum()}, Total: {len(all_doc_ids)}")

    return {
        "S_full": S_full,
        "S_sec": S_sec,
        "all_doc_ids": all_doc_ids,
        "sec_doc_ids": sec_doc_ids,
        "sec_indices": sec_indices,
        "sec_mask": sec_mask,
        "chi_mask": chi_mask,
        "gt_5class": gt_5class,
        "gt_4class": gt_4class,
        "representations": representations,
    }
