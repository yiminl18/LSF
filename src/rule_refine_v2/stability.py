"""Stage F — k-fold stability filter.

Run Stages B-D on k random subsets of the sampled docs and keep rules selected
in at least ⌈k/2⌉ folds. This filters out rules that look good only on a single
layout idiosyncrasy.

Stage A is shared across folds (computed once on all sampled docs); the folds
re-run only B-D, which are LLM-free.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from .set_cover import greedy_cover
from .target_docs import build_target_set


def _fold_profile(profile: dict[str, dict], doc_subset: set[str]) -> dict[str, dict]:
    """Restrict each rule's covered_docs and proxy_docs to a doc subset."""
    new_profile: dict[str, dict] = {}
    n = max(1, len(doc_subset))
    for r, p in profile.items():
        covered = p["covered_docs"] & doc_subset
        proxy   = p["proxy_docs"]   & doc_subset
        new_profile[r] = {
            **p,
            "covered_docs": covered,
            "proxy_docs":   proxy,
            "cov":          round(len(covered) / n, 4),
            "proxy_acc":    round(len(proxy)   / n, 4),
        }
    return new_profile


def k_fold_selection(
    profile: dict[str, dict],
    all_doc_names: set[str],
    *,
    k_folds: int = 5,
    fold_frac: float = 0.7,
    alpha: float = 1.0,
    beta:  float = 1.0,
    eps:   float = 1e-4,
    seed: int | None = 42,
) -> tuple[list[dict], Counter]:
    """Run k random folds and return (fold_results, selection_freq).

    fold_results: list of {fold, docs, D_star, selected}
    selection_freq: Counter mapping rule name → number of folds it appeared in.
    """
    if seed is not None:
        random.seed(seed)

    docs_list = sorted(all_doc_names)
    fold_size = max(2, int(len(docs_list) * fold_frac))
    freq: Counter = Counter()
    fold_results: list[dict] = []

    for fold in range(k_folds):
        subset = set(random.sample(docs_list, fold_size))
        sub_profile = _fold_profile(profile, subset)
        D_star_fold = build_target_set(sub_profile)
        selected, _ = greedy_cover(sub_profile, D_star_fold, alpha=alpha, beta=beta, eps=eps)
        freq.update(selected)
        fold_results.append({
            "fold":     fold,
            "docs":     sorted(subset),
            "D_star":   sorted(D_star_fold),
            "selected": selected,
        })

    return fold_results, freq


def stable_rule_set(
    freq: Counter,
    base_selection: list[str],
    *,
    k_folds: int,
    min_folds: int | None = None,
) -> list[str]:
    """Return rules with freq ≥ min_folds, preserving the base_selection order.

    Rules in base_selection that don't meet the threshold are dropped.
    Rules outside base_selection that meet the threshold are appended at the end
    in descending freq order (these are "robust additions").
    """
    threshold = min_folds if min_folds is not None else math.ceil(k_folds / 2)
    kept = [r for r in base_selection if freq.get(r, 0) >= threshold]
    extras = [r for r, c in freq.most_common()
              if c >= threshold and r not in kept and r not in base_selection]
    return kept + extras
