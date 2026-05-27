"""Phase 3: per-rule tau coverage check.

cov(r) is the fraction of sampled docs that rule r alone answers correctly.
Rules with cov(r) < tau are banned from the selected set; the docs they were
responsible for are returned for re-coverage by Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load_or_compute_coverage(
    rule_name: str,
    eval_individual_path: Path,
    fallback: Callable[[], float],
) -> float:
    """Return cov(r): read accuracy from disk if the eval file exists, else call fallback."""
    if eval_individual_path.exists():
        return json.loads(eval_individual_path.read_text(encoding="utf-8"))["accuracy"]
    return fallback()


def filter_by_tau(
    selected_rules: list[str],
    per_rule_gained: dict[str, set[str]],
    eval_individual_dir: Path,
    tau: float,
    fallback_fn: Callable[[str], float] | None = None,
) -> tuple[list[str], set[str]]:
    """Remove rules with cov(r) < tau from selected_rules.

    Returns:
        filtered_rules: selected_rules with low-coverage rules removed.
        docs_to_recover: union of gained-doc sets for banned rules (need re-coverage).
    """
    banned: list[str] = []
    docs_to_recover: set[str] = set()

    for r in list(selected_rules):
        eval_path = eval_individual_dir / f"{r}_eval.json"
        fb: Callable[[], float] = (lambda r_=r: fallback_fn(r_)) if fallback_fn else (lambda: 0.0)
        cov = load_or_compute_coverage(r, eval_path, fb)
        if cov < tau:
            banned.append(r)
            docs_to_recover |= per_rule_gained.get(r, set())
            print(f"  BAN {r}  cov={cov:.3f} < tau={tau:.3f}")

    return [r for r in selected_rules if r not in banned], docs_to_recover
