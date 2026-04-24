"""core.cluster.bisection.llm_merge — Corpus-level LLM cluster merging."""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from core.cluster.common.paths import SEED
from core.llm.model import llm_call

logger = logging.getLogger(__name__)


def sample_cluster_headings(
    doc_ids: list[str],
    rep: dict,
    max_docs: int = 10,
    max_headings_per_doc: int = 8,
) -> str:
    """Sample representative heading text from a cluster for LLM judgment.
    
    Deduplicates and selects a subset of headings across multiple documents 
    to provide a concise semantic profile of the cluster within prompt limits.
    """
    rng = np.random.RandomState(SEED)
    if len(doc_ids) > max_docs:
        sampled = rng.choice(doc_ids, max_docs, replace=False).tolist()
    else:
        sampled = list(doc_ids)

    all_headings: list[str] = []
    for d in sampled:
        texts = [n["text"] for n in rep[d]["nodes"] if n.get("text")]
        all_headings.extend(texts[:max_headings_per_doc])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in all_headings:
        h_lower = h.strip().lower()
        if h_lower not in seen:
            seen.add(h_lower)
            unique.append(h.strip())

    return "\n".join(f'- "{h}"' for h in unique[:20])


def llm_merge_clusters(
    assignments: dict[str, str],
    rep: dict,
    llm_provider: str = "azure",
    *,
    llm_model: str,
) -> dict[str, str]:
    """Run one corpus-level LLM pass to merge over-split clusters.
    
    Leverages LLM reasoning to identify clusters that share the same high-level 
    document type (e.g., merging different '10-Q' variants) based on their 
    sampled heading patterns.
    """
    clusters: dict[str, list[str]] = {}
    for doc_id, label in assignments.items():
        clusters.setdefault(label, []).append(doc_id)
    sorted_labels = sorted(clusters.keys())

    # Build prompt
    sections = []
    for i, label in enumerate(sorted_labels):
        doc_ids = clusters[label]
        headings = sample_cluster_headings(doc_ids, rep)
        sections.append(f"Cluster {i} ({len(doc_ids)} docs):\n{headings}")

    prompt = (
        f"Below are representative heading texts from {len(sorted_labels)} "
        "document clusters.\n\n"
        + "\n\n".join(sections)
        + "\n\nBased on heading text patterns, identify which clusters represent "
        "the SAME type of document and should be merged. "
        "Output ONLY a JSON list of merge groups, where each group is a list of "
        "cluster indices that should be combined.\n"
        "Example: [[0, 3], [1, 4, 5], [2]]\n"
        "Merge only when heading patterns clearly indicate the same document type. "
        "When in doubt, keep separate."
    )

    logger.info(f"    LLM merge: {len(sorted_labels)} clusters → LLM...")
    response = llm_call(
        prompt,
        llm_provider=llm_provider,
        model=llm_model,
        max_tokens=200,
    )
    logger.info(f"    LLM response: {response.strip()}")

    # Parse JSON
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if not json_match:
        logger.warning("    LLM merge: failed to parse JSON, keeping original assignments")
        return assignments

    try:
        merge_groups = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("    LLM merge: JSON decode failed, keeping original assignments")
        return assignments

    # Execute merge
    new_assignments = dict(assignments)
    for group in merge_groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        target_label = sorted_labels[group[0]]
        for idx in group[1:]:
            if 0 <= idx < len(sorted_labels):
                source_label = sorted_labels[idx]
                for doc_id in clusters.get(source_label, []):
                    new_assignments[doc_id] = target_label

    merged_k = len(set(new_assignments.values()))
    logger.info(f"    LLM merge: {len(sorted_labels)} → {merged_k} clusters")
    return new_assignments



__all__ = [
    "sample_cluster_headings",
    "llm_merge_clusters",
]
