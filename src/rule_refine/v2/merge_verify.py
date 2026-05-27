"""Stage E — merge accuracy verification with the real LLM judge.

Wraps `rule_refine.evaluate_merge_accuracy` so the v2 selector can verify the
greedy-cover output against the real QA + judge pipeline (not the proxy).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_refine.v1 import evaluate_merge_accuracy  # noqa: E402


def verify_and_extend(
    selected: list[str],
    ranked_pool: list[tuple[str, float]],
    documents: list[dict],
    ground_truth: dict,
    question: str,
    rule_folder: Path,
    model_mod,
    *,
    target_accuracy: float,
    max_extra_rules: int = 5,
) -> tuple[list[str], float, list[dict], dict]:
    """Verify selected against the LLM judge; add rules by utility if below target.

    Returns (final_selected, merge_accuracy, per_doc, token_accounting).
    """
    current = list(selected)
    in_set  = set(current)
    pool_iter = (r for r, _ in ranked_pool if r not in in_set)

    tokens = {"qa_in": 0, "qa_out": 0, "judge_in": 0, "judge_out": 0, "llm_calls": 0}

    acc, per_doc, qa_in, qa_out, j_in, j_out = evaluate_merge_accuracy(
        current, documents, ground_truth, question, rule_folder, model_mod
    )
    tokens["qa_in"]     += qa_in
    tokens["qa_out"]    += qa_out
    tokens["judge_in"]  += j_in
    tokens["judge_out"] += j_out
    tokens["llm_calls"] += 2 * len(documents)

    extras_added: list[dict] = []
    iters = 0
    while acc < target_accuracy and iters < max_extra_rules:
        try:
            next_r = next(pool_iter)
        except StopIteration:
            break
        current.append(next_r)
        in_set.add(next_r)
        iters += 1

        acc, per_doc, qa_in, qa_out, j_in, j_out = evaluate_merge_accuracy(
            current, documents, ground_truth, question, rule_folder, model_mod
        )
        tokens["qa_in"]     += qa_in
        tokens["qa_out"]    += qa_out
        tokens["judge_in"]  += j_in
        tokens["judge_out"] += j_out
        tokens["llm_calls"] += 2 * len(documents)
        extras_added.append({
            "iter":     iters,
            "added":    next_r,
            "acc_after": round(acc, 4),
        })

    return current, acc, per_doc, {**tokens, "extras": extras_added}
