"""Shared diversity utilities for tool-agent variants.

Two helpers consumed by `diverse_core` (DIVERSITY GATE per turn) and by
`seq_cover_core` (signature gate inside `_run_single_rule_episode` to reject
duplicate rules and re-prompt the LLM):

- `_max_chars_bucket(max_chars)` — coarse 3-bucket categorisation
  (small ≤200, medium ≤800, large >800).
- `_rule_diversity_signature(rule)` — `(mode, max_chars_bucket, anchor[:50] casefold)`.
  Two rules with the same signature are considered "duplicates"; the second is
  rejected. The `[:50]` window is empirically tuned (see commit history of
  diverse_core.py): `[:30]` let near-duplicate anchors pass on q3, `[:50]`
  balances substring overlap detection vs over-aggressive family merging.
"""

from __future__ import annotations

from agent.rules.range_rule_json import RangeRule


def _max_chars_bucket(max_chars: int) -> str:
    """3-bucket coarse categorisation for diversity hashing."""
    if max_chars <= 200:
        return "small"
    if max_chars <= 800:
        return "medium"
    return "large"


def _rule_diversity_signature(rule: RangeRule) -> tuple[str, str, str]:
    """Hash a rule into (mode, max_chars_bucket, anchor[:50] casefold) for diversity check.

    Two rules with the same signature are considered "duplicates" and the second is
    rejected. anchor[:50] balances substring overlap detection vs over-aggressive
    family merging ([:30] let near-dup anchors pass on q3).
    """
    spec = rule.retrieval_spec
    return (
        spec.mode,
        _max_chars_bucket(spec.max_chars),
        ((spec.anchor or "")[:50].casefold()),
    )
