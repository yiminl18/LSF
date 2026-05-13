"""Adapter from ExtractionResult to DeployedRow-shaped dict.

Calls score_generated_answer from range_rule_scorer, wraps the result
into a dict whose key set matches rule_runtime.deploy.DeployedRow.
"""

from __future__ import annotations

from typing import Any

from agent.baselines.base import ExtractionResult
from agent.rule_runtime.deploy import DeployedRow
from agent.rules.range_rule_scorer import score_generated_answer, normalize_answer_text
from core.pipeline.e2e_utils.cache import CachedLLMCaller


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
    equivalent for baselines; they are set to empty values. The trace
    is stored in 'blocker' only when an error is present.
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
        retrieved_subset_text="",
        generated_answer=result.generated_answer,
        judge_result=score.judge_result,
        blocker=None if score.judge_result else "judge_false",
        gen_calls=1,
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
