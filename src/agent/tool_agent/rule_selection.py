"""Rule selection — greedy set-cover on the (rule, success_doc) matrix.

Replaces the top-K-by-score approach to reduce Phase B cost:
- Two rules that hit the exact same docs on sampled evaluation contribute nearly zero
  marginal value in Phase B, but top-K-by-score would send both, wasting
  (rule × holdout_docs) LLM evaluations.
- Greedy set-cover (Chvátal 1979) gives the classic (1-1/e) ≈ 0.63 approximation ratio
  and naturally maps to "which rule adds new coverage to the union accuracy" in our setting
  where elements = correctly-answered sampled docs.
"""

from __future__ import annotations

from typing import Any

# Lower bound on per-rule accuracy on sampled docs. Rules below this threshold are
# treated as unreliable on the sample and excluded from Phase B.
# 0.5 is a conservative starting point (just above random baseline).
TAU_ACC: float = 0.5

# Tie-break tolerance: when marginal_gain == 0 but a rule's score is close to the
# top-selected rule (gap < DELTA), keep it for ensemble diversity. Avoids over-aggressive
# pruning of near-equal rules under sampled noise.
DELTA: float = 0.05

# Minimum ensemble size: when set-cover selects fewer than MIN_SELECTED rules, pad up
# to MIN_SELECTED by score desc. An ensemble buffer stabilises holdout performance
# much better than a single rule given anchor variability across documents.
MIN_SELECTED: int = 3


def greedy_set_cover(
    cross_doc_eval: list[dict[str, Any]],
    max_rules: int = 5,
    tau_acc: float = TAU_ACC,
    delta: float = DELTA,
) -> list[int]:
    """Greedily select the coverage-maximising rule subset from cross_doc_eval.

    cross_doc_eval is assumed to be sorted by score desc (orchestrator guarantees this).
    Each entry must contain `success_doc_ids: list[str]`.

    Args:
        cross_doc_eval: Output of orchestrator._cross_doc_evaluate.
        max_rules: Hard upper bound on selected rules (controls Phase B cost).
        tau_acc: Minimum accuracy threshold; rules below this are excluded.
        delta: Tie-break tolerance for zero-gain rules with near-top scores.

    Returns:
        List of selected rule indices in insertion order; empty list on failure
        (caller falls back to top-K).
    """
    # Filter: tau_acc gate + must have success_doc_ids + skip budget-skipped entries
    candidates = [
        e for e in cross_doc_eval
        if not e.get("skipped_due_to_budget", False)
        and e.get("accuracy", 0.0) >= tau_acc
        and e.get("success_doc_ids")  # at least one doc judge-passed
    ]
    if not candidates:
        return []

    covered: set[str] = set()
    selected: list[int] = []
    top_score = candidates[0]["score"]  # candidates already sorted by score desc

    while candidates and len(selected) < max_rules:
        # argmax marginal_gain; ties resolved by score order (list ordering guarantees first = highest)
        best_idx = 0
        best_gain = len(set(candidates[0]["success_doc_ids"]) - covered)
        for i in range(1, len(candidates)):
            gain = len(set(candidates[i]["success_doc_ids"]) - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        chosen = candidates.pop(best_idx)

        if best_gain >= 1:
            selected.append(chosen["rule_index"])
            covered |= set(chosen["success_doc_ids"])
            continue

        # marginal_gain == 0: keep rule if its score is near the top (ensemble diversity)
        if (top_score - chosen["score"]) < delta and selected:
            selected.append(chosen["rule_index"])
            continue

        # Pad up to MIN_SELECTED: set-cover saying "no new coverage" on sampled docs
        # does not mean the rule has no value on holdout docs.
        if len(selected) < MIN_SELECTED:
            selected.append(chosen["rule_index"])
            continue

        # Already at minimum ensemble; rule adds no coverage and score is not near top — stop
        break

    return selected


def select_best_rules(
    cross_doc_eval: list[dict[str, Any]],
    unique_rules: list[Any],  # list[RangeRule]; Any here to avoid circular import
    max_rules: int = 5,
    *,
    allow_no_score_first_three_fallback: bool = True,
) -> list[Any]:
    """Public interface: cross_doc_eval + unique_rules → selected RangeRule list.

    Prefers set-cover; falls back to top-K by score when set-cover returns nothing
    (preserving historical orchestrator behaviour when caller allows it).
    """
    selected_indices = greedy_set_cover(cross_doc_eval, max_rules=max_rules)

    if selected_indices:
        return [unique_rules[i] for i in selected_indices]

    # Fallback: top-K with score > 0, or first 3 if none qualify
    fallback = [
        unique_rules[e["rule_index"]] for e in cross_doc_eval if e.get("score", 0) > 0
    ][:max_rules]
    # Historical RangeRule callers keep the first-three rescue; stricter callers
    # can disable it without rule-kind coupling in this selector.
    if not fallback and unique_rules and allow_no_score_first_three_fallback:
        fallback = unique_rules[:3]
    return fallback
