"""Phase 4 (optional): drop and swap tests to trim the selected rule set.

drop_test: for each r ∈ S, check if removing r leaves all credited docs still
           correct.  If so, drop r.

swap_test: find the most expensive r ∈ S and the cheapest r' ∉ S.  Admit the
           swap iff coverage on all of D* is preserved and total cost strictly
           decreases.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_apply_merge import rule_apply_merge
from rule_refinement.eval_judge import judge

_DEFAULT_OUTPUT_DIR = (
    "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selector_run"
)


def drop_test(
    selected_rules: list[str],
    per_rule_gained: dict[str, set[str]],
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    rules_dir: str,
    labels: dict,
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """Return a trimmed copy of selected_rules with redundant rules removed.

    A rule is redundant if all docs it was credited with remain correct when
    it is absent from the set.
    """
    result = list(selected_rules)

    for r in list(result):
        without_r = [x for x in result if x != r]
        credited = per_rule_gained.get(r, set())

        if not credited:
            result.remove(r)
            print(f"  DROP {r} (no credited docs)")
            continue

        still_all_correct = True
        for d in credited:
            res = rule_apply_merge(
                document=documents[d],
                rule_names=without_r,
                question_slug=question_slug,
                question=question,
                model_name=model_name,
                rules_dir=rules_dir,
                output_dir=output_dir,
            )
            gt = labels.get(d + ".pdf", {}).get(question)
            correct, _, _ = judge(question, gt, res["predicted_answer"], model_name=model_name)
            if not correct:
                still_all_correct = False
                break

        if still_all_correct:
            result.remove(r)
            print(f"  DROP {r} (redundant)")

    return result


def swap_test(
    selected_rules: list[str],
    cost_profile: dict[str, dict],
    candidate_pool: list[str],
    target_docs: set[str],
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    rules_dir: str,
    labels: dict,
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """Try replacing the most expensive rule with the cheapest candidate outside S.

    The swap is admitted only if coverage on all of target_docs is preserved
    and the total avg_cost_ratio strictly decreases.
    """
    if not selected_rules:
        return selected_rules

    S_set = set(selected_rules)
    available = [r for r in candidate_pool if r not in S_set]
    if not available:
        return selected_rules

    most_expensive = max(
        selected_rules,
        key=lambda r: cost_profile.get(r, {}).get("avg_cost_ratio", 0.0),
    )
    cheapest_candidate = min(
        available,
        key=lambda r: cost_profile.get(r, {}).get("avg_cost_ratio", float("inf")),
    )

    current_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in selected_rules)
    swapped = [r for r in selected_rules if r != most_expensive] + [cheapest_candidate]
    swapped_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in swapped)

    if swapped_cost >= current_cost:
        return selected_rules

    for d in target_docs:
        res = rule_apply_merge(
            document=documents[d],
            rule_names=swapped,
            question_slug=question_slug,
            question=question,
            model_name=model_name,
            rules_dir=rules_dir,
            output_dir=output_dir,
        )
        gt = labels.get(d + ".pdf", {}).get(question)
        correct, _, _ = judge(question, gt, res["predicted_answer"], model_name=model_name)
        if not correct:
            return selected_rules

    print(
        f"  SWAP {most_expensive} → {cheapest_candidate}  "
        f"cost {current_cost:.5f} → {swapped_cost:.5f}"
    )
    return swapped
