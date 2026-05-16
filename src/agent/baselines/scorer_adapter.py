"""Adapter from ExtractionResult to DeployedRow-shaped dict.

Calls score_generated_answer from range_rule_scorer, wraps the result
into a dict whose key set matches rule_runtime.deploy.DeployedRow.
"""

from __future__ import annotations

import json

from agent.baselines.base import ExtractionResult
from agent.rule_runtime.deploy import DeployedRow
from agent.rules.range_rule_scorer import score_generated_answer, normalize_answer_text
from core.pipeline.e2e_utils.cache import CachedLLMCaller


def _blocker_from_judge(judge_result: bool | str) -> str | None:
    """Return None for a True judge, a specific tag otherwise.

    A non-bool judge value (e.g. "unknown") gets its own tag so downstream
    filtering can distinguish it from an unambiguous "judge said false".
    """
    if judge_result is True:
        return None
    if isinstance(judge_result, bool):
        return "judge_false"
    return f"judge_{judge_result!s}"


def _trace_snippet(trace: object) -> str:
    """Compact JSON-safe rendering of an ExtractionResult.trace for the row."""
    try:
        return json.dumps(trace, ensure_ascii=False, default=str)[:4000]
    except (TypeError, ValueError):
        return str(trace)[:4000]


def score_and_build_row(
    *,
    query_idx: int,
    doc_id: str,
    policy: str,
    result: ExtractionResult,
    ground_truth: str,
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> DeployedRow:
    """Score an ExtractionResult and return a DeployedRow-shaped dict.

    The 'rules_used' and 'retrieved_subset_text' fields have no direct
    equivalent for baselines. We keep retrieved_subset_text empty and stash a
    compact trace snippet in 'blocker' only when an error or judge failure is
    present (DeployedRow has no dedicated trace field).
    """
    normalized_gt = normalize_answer_text(ground_truth)

    score = score_generated_answer(
        question=query_text,
        generated_answer=result.generated_answer,
        ground_truth=normalized_gt,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    judge_cost = score.metadata.get("judge", {}).get("cost_usd", 0.0)
    total_cost = result.cost_usd + judge_cost

    return DeployedRow(
        query_idx=query_idx,
        doc_id=doc_id,
        policy=policy,
        rules_used=[],
        retrieved_subset_text=_trace_snippet(result.trace),
        generated_answer=result.generated_answer,
        judge_result=score.judge_result,
        blocker=_blocker_from_judge(score.judge_result),
        gen_calls=int(getattr(result, "gen_calls", 1) or 1),
        judge_calls=1,
        actual_cost_usd=total_cost,
    )


def error_row(
    *,
    query_idx: int,
    doc_id: str,
    policy: str,
    blocker: str,
) -> DeployedRow:
    """Build an empty DeployedRow for an extraction that failed before scoring."""
    return DeployedRow(
        query_idx=query_idx,
        doc_id=doc_id,
        policy=policy,
        rules_used=[],
        retrieved_subset_text="",
        generated_answer=None,
        judge_result=False,
        blocker=blocker,
        gen_calls=0,
        judge_calls=0,
        actual_cost_usd=0.0,
    )
