"""Sequential-covering tool-agent: generate a rule, claim covered docs, repeat."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace as dc_replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.rules.range_rule_exec import execute_range_rule
from agent.rules.range_rule_json import RangeRule
from agent.rules.range_rule_scorer import score_retrieved_subset
from agent.tool_agent.core import (
    AgentConfig,
    AgentResult,
    _build_corpus_toc,
    _compact_history,
    _ConversationTurn,
    _detect_stale_anchor,
    _serialize_prompt,
    _truncate_observation,
)
from agent.tool_agent.diverse_core import _parse_rule_payload
from agent.tool_agent.diversity import _max_chars_bucket, _rule_diversity_signature
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from agent.prompts.tool_agent.tool_agent_seq_cover_action_schema import (
    _build_seq_cover_action_schema,
)
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.cache import CachedLLMCaller

_DECODER = json.JSONDecoder()
_SEQ_COVER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "tool_agent"
    / "tool_agent_system_seq_cover.txt"
)
_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_seq_cover_template() -> str:
    return _SEQ_COVER_PROMPT_PATH.read_text(encoding="utf-8")


def _build_seq_cover_system_prompt(
    query_text: str,
    corpus_toc: str,
    uncovered_doc_ids: list[str],
    frozen_summary: str,
) -> str:
    template = _load_seq_cover_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{uncovered_doc_ids_json}", json.dumps(uncovered_doc_ids, ensure_ascii=False))
        .replace("{frozen_summary}", frozen_summary or "(none yet)")
    )


def _format_frozen_summary(rules: list[RangeRule]) -> str:
    if not rules:
        return ""
    lines: list[str] = []
    for idx, rule in enumerate(rules):
        spec = rule.retrieval_spec
        lines.append(
            f"  F{idx}: mode={spec.mode} bucket={_max_chars_bucket(spec.max_chars)} "
            f"max_chars={spec.max_chars} anchor={(spec.anchor or '')[:60]!r}"
        )
    return "\n".join(lines)


def _run_single_rule_episode(
    *,
    query_text: str,
    query_idx: int,
    iteration_idx: int,
    window_contexts: list[DocumentContext],
    frozen_rules: list[RangeRule],
    uncovered_doc_ids: list[str],
    existing_signatures: set[tuple[str, str, str]],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
) -> AgentResult:
    corpus_toc = _build_corpus_toc(window_contexts)
    action_schema = _build_seq_cover_action_schema()
    history: list[_ConversationTurn] = []
    total_cost = 0.0
    registry = ToolRegistry(window_contexts[0], peer_docs=window_contexts[1:])
    corpus_id_label = f"corpus[{len(window_contexts)}](seq_cover:i{iteration_idx})"
    nav_turn_counter = 0
    dup_rejects_this_episode = 0
    force_distinct = bool(getattr(agent_config, "seq_cover_force_distinct_signature", True))
    max_signature_retries = int(getattr(agent_config, "seq_cover_max_signature_retries", 2))

    for turn_index in range(agent_config.max_turns_per_query):
        if total_cost >= agent_config.budget_usd:
            return AgentResult(
                rules=[],
                trajectory=[],
                total_cost_usd=total_cost,
                turns_used=turn_index,
                termination_reason="budget",
            )

        compact_hist = _compact_history(
            history,
            keep_recent=agent_config.history_compaction_threshold,
        )
        system_prompt = _build_seq_cover_system_prompt(
            query_text=query_text,
            corpus_toc=corpus_toc,
            uncovered_doc_ids=uncovered_doc_ids,
            frozen_summary=_format_frozen_summary(frozen_rules),
        )
        stale_warning = _detect_stale_anchor(history, agent_config.stale_anchor_threshold)
        current_system = system_prompt + (stale_warning or "")
        if nav_turn_counter >= agent_config.nav_turn_cap:
            current_system += (
                f"\n\n[system] Navigation budget exhausted "
                f"({agent_config.nav_turn_cap} consecutive find_section calls). "
                "Next action MUST be `generate` or `batch_apply_rule`."
            )
        prompt = _serialize_prompt(current_system, compact_hist)

        try:
            _ensure_request_within_model_context(
                prompt_text=prompt,
                max_output_tokens=agent_config.agent_max_tokens,
                llm_provider=agent_config.agent_llm_provider,
                llm_model=agent_config.agent_llm_model,
                stage_label=f"seq_cover_agent q{query_idx} iter={iteration_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            _LOG.warning(
                "seq-cover prompt over context limit: q%s iter=%s turn=%s error=%s",
                query_idx,
                iteration_idx,
                turn_index,
                exc,
            )
            return AgentResult(
                rules=[],
                trajectory=[],
                total_cost_usd=total_cost,
                turns_used=turn_index,
                termination_reason="context_limit",
            )

        t0 = time.time()
        try:
            cache_result = cached_caller.call(
                prompt=prompt,
                llm_provider=agent_config.agent_llm_provider,
                max_tokens=agent_config.agent_max_tokens,
                model=agent_config.agent_llm_model,
                response_schema=action_schema,
            )
        except Exception as exc:
            _LOG.warning(
                "seq-cover LLM call failed: q%s iter=%s turn=%s error=%s",
                query_idx,
                iteration_idx,
                turn_index,
                exc,
            )
            break

        agent_latency = (time.time() - t0) * 1000
        agent_call_cost = compute_cost(
            cache_result.input_tokens,
            cache_result.output_tokens,
            agent_config.agent_llm_provider,
            model=agent_config.agent_llm_model,
        )
        total_cost += agent_call_cost

        try:
            action, _end = _DECODER.raw_decode(cache_result.response.lstrip())
        except json.JSONDecodeError:
            obs = "ERROR: invalid JSON response"
            history.append(
                _ConversationTurn(
                    turn_index=turn_index,
                    action_json=cache_result.response[:500],
                    tool_name=None,
                    observation=obs,
                )
            )
            logger.log_turn(
                query_idx=query_idx,
                doc_id=corpus_id_label,
                turn_index=turn_index,
                agent_reasoning="",
                tool_name="invalid_json",
                tool_args={},
                tool_result_preview=obs,
                cost_usd=agent_call_cost,
                latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=iteration_idx,
                tool_result_full={"error": "invalid_json", "iteration_idx": iteration_idx},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

        action_type = action.get("action", "")
        reasoning = action.get("reasoning", "")
        if action_type == "generate":
            new_rule = _parse_rule_payload(action.get("rule"))
            if new_rule is None:
                obs = "ERROR: 'rule' missing or malformed; provide exactly one valid retrieval_spec."
                history.append(
                    _ConversationTurn(
                        turn_index=turn_index,
                        action_json=json.dumps(action, ensure_ascii=False),
                        tool_name="generate",
                        observation=obs,
                    )
                )
                logger.log_turn(
                    query_idx=query_idx,
                    doc_id=corpus_id_label,
                    turn_index=turn_index,
                    agent_reasoning=reasoning,
                    tool_name="generate_invalid",
                    tool_args={"iteration_idx": iteration_idx},
                    tool_result_preview=obs,
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=iteration_idx,
                    tool_result_full={"error": "invalid_rule_payload"},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            sig = _rule_diversity_signature(new_rule)
            if force_distinct and sig in existing_signatures:
                dup_rejects_this_episode += 1
                obs = (
                    f"REJECTED: rule duplicates an Already-Selected rule on diversity "
                    f"signature {sig!r}. Vary `mode` (after/before/around/between/page/regex) "
                    f"or `max_chars` bucket (small ≤200, medium ≤800, large >800) or "
                    f"`anchor` (first 50 chars). Already-selected signatures: "
                    f"{sorted(existing_signatures)!r}."
                )
                history.append(
                    _ConversationTurn(
                        turn_index=turn_index,
                        action_json=json.dumps(action, ensure_ascii=False),
                        tool_name="generate",
                        observation=obs,
                    )
                )
                logger.log_turn(
                    query_idx=query_idx,
                    doc_id=corpus_id_label,
                    turn_index=turn_index,
                    agent_reasoning=reasoning,
                    tool_name="generate_rejected_duplicate",
                    tool_args={"signature": list(sig), "iteration_idx": iteration_idx},
                    tool_result_preview=obs,
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=iteration_idx,
                    tool_result_full={"rejected": True, "signature": list(sig)},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                if dup_rejects_this_episode > max_signature_retries:
                    break
                continue

            obs = f"Rule accepted for sequential-cover iteration {iteration_idx}."
            logger.log_turn(
                query_idx=query_idx,
                doc_id=corpus_id_label,
                turn_index=turn_index,
                agent_reasoning=reasoning,
                tool_name="generate_accepted",
                tool_args={"iteration_idx": iteration_idx},
                tool_result_preview=obs,
                cost_usd=agent_call_cost,
                latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=iteration_idx,
                tool_result_full={"rule": new_rule.to_dict(), "iteration_idx": iteration_idx},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            return AgentResult(
                rules=[new_rule],
                trajectory=[],
                total_cost_usd=total_cost,
                turns_used=turn_index + 1,
                termination_reason="generate",
            )

        tool_name = action.get("tool", "")
        raw_args = action.get("args", "{}")
        try:
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            tool_args = {}

        tool_result = registry.dispatch(tool_name, tool_args)
        total_cost += tool_result.cost_usd
        if tool_name == "find_section":
            nav_turn_counter += 1
        elif tool_name == "batch_apply_rule":
            nav_turn_counter = 0
        if tool_result.success:
            obs_text = json.dumps(tool_result.data, ensure_ascii=False, default=str)
        else:
            obs_text = f"ERROR: {tool_result.error}"
        obs_text = _truncate_observation(obs_text, agent_config.observation_max_chars)
        history.append(
            _ConversationTurn(
                turn_index=turn_index,
                action_json=json.dumps(action, ensure_ascii=False),
                tool_name=tool_name,
                observation=obs_text,
            )
        )
        logger.log_turn(
            query_idx=query_idx,
            doc_id=corpus_id_label,
            turn_index=turn_index,
            agent_reasoning=reasoning,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result_preview=obs_text[:500],
            cost_usd=agent_call_cost + tool_result.cost_usd,
            latency_ms=agent_latency + tool_result.latency_ms,
            input_tokens=cache_result.input_tokens,
            output_tokens=cache_result.output_tokens,
            path_idx=iteration_idx,
            tool_result_full=tool_result.data if tool_result.success else {"error": tool_result.error},
            prompt_text=prompt,
            raw_response=cache_result.response,
        )

    return AgentResult(
        rules=[],
        trajectory=[],
        total_cost_usd=total_cost,
        turns_used=agent_config.max_turns_per_query,
        termination_reason="max_turns",
    )


def _evaluate_rule_on_remaining(
    rule: RangeRule,
    remaining_contexts: list[DocumentContext],
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> tuple[list[str], list[dict[str, Any]], float]:
    covered: list[str] = []
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for doc in remaining_contexts:
        subset = execute_range_rule(rule, doc.normalized_text)
        span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
        row: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "matched": bool(subset.matched),
            "retrieved_subset_chars": len(span_text),
            "judge_result": "NOT_RUN",
        }
        if subset.matched and span_text.strip():
            score = score_retrieved_subset(
                question=query_text,
                retrieved_text=span_text,
                ground_truth=doc.ground_truth,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            total_cost += (
                score.metadata["generation"]["cost_usd"]
                + score.metadata["judge"]["cost_usd"]
            )
            row["generated_answer"] = score.generated_answer
            row["judge_result"] = score.judge_result
            if score.judge_result:
                covered.append(doc.doc_id)
        rows.append(row)
    return covered, rows, total_cost


def _write_seq_cover_trajectory(logger: TrajectoryLogger, payload: dict[str, Any]) -> None:
    output_dir = logger.output_dir
    (output_dir / "seq_cover_trajectory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_seq_cover_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Generate-cover-remove over sampled documents."""
    _ = path_idx  # Reserved for API compatibility with sibling agent modes.
    if not doc_contexts:
        return AgentResult(
            rules=[],
            trajectory=[],
            total_cost_usd=0.0,
            turns_used=0,
            termination_reason="no_docs",
        )

    max_rules = int(getattr(agent_config, "seq_cover_max_rules", 6))
    max_iterations = int(getattr(agent_config, "seq_cover_max_iterations", max_rules))
    min_coverage = int(getattr(agent_config, "seq_cover_min_coverage_threshold", 1))
    min_marginal = int(getattr(agent_config, "seq_cover_min_marginal_coverage", 1))
    uncovered_window = int(getattr(agent_config, "seq_cover_uncovered_doc_window", 4))
    patience = int(getattr(agent_config, "seq_cover_patience", 2))
    budget_factor = float(getattr(agent_config, "seq_cover_per_iteration_budget_factor", 1.0))
    iteration_turns = (
        max_turns
        if max_turns is not None
        else int(getattr(agent_config, "seq_cover_per_iteration_max_turns", 4))
    )
    iteration_config = dc_replace(
        agent_config,
        max_turns_per_query=max(1, iteration_turns),
        budget_usd=max(0.0, agent_config.budget_usd * budget_factor),
    )

    id_to_ctx = {doc.doc_id: doc for doc in doc_contexts}
    uncovered_doc_ids = [doc.doc_id for doc in doc_contexts]
    frozen_rules: list[RangeRule] = []
    iterations: list[dict[str, Any]] = []
    total_cost = 0.0
    total_turns = 0
    non_progress_count = 0
    termination_reason = "max_iterations"

    for iteration_idx in range(max_iterations):
        if not uncovered_doc_ids:
            termination_reason = "full_coverage"
            break
        if len(frozen_rules) >= max_rules:
            termination_reason = "max_rules"
            break
        if total_cost >= agent_config.budget_usd:
            termination_reason = "budget"
            break

        window_ids = uncovered_doc_ids[: max(1, uncovered_window)]
        window_contexts = [id_to_ctx[doc_id] for doc_id in window_ids]
        existing_signatures = {_rule_diversity_signature(r) for r in frozen_rules}
        episode = _run_single_rule_episode(
            query_text=query_text,
            query_idx=query_idx,
            iteration_idx=iteration_idx,
            window_contexts=window_contexts,
            frozen_rules=frozen_rules,
            uncovered_doc_ids=window_ids,
            existing_signatures=existing_signatures,
            cached_caller=cached_caller,
            agent_config=iteration_config,
            logger=logger,
        )
        total_cost += episode.total_cost_usd
        total_turns += episode.turns_used

        if not episode.rules:
            non_progress_count += 1
            iterations.append(
                {
                    "iteration_idx": iteration_idx,
                    "accepted": False,
                    "coverage_set": [],
                    "remaining_size": len(uncovered_doc_ids),
                    "termination_reason": episode.termination_reason,
                }
            )
            if non_progress_count >= patience:
                termination_reason = "patience"
                break
            continue

        candidate = episode.rules[0]
        remaining_contexts = [id_to_ctx[doc_id] for doc_id in uncovered_doc_ids]
        coverage_set, eval_rows, eval_cost = _evaluate_rule_on_remaining(
            candidate,
            remaining_contexts,
            query_text,
            cached_caller,
            agent_config.agent_llm_provider,
            agent_config.agent_llm_model,
        )
        total_cost += eval_cost
        marginal = len(coverage_set)
        accepted = marginal >= min_coverage and marginal >= min_marginal
        if accepted:
            frozen_rules.append(candidate)
            covered = set(coverage_set)
            uncovered_doc_ids = [doc_id for doc_id in uncovered_doc_ids if doc_id not in covered]
            non_progress_count = 0
        else:
            non_progress_count += 1

        iterations.append(
            {
                "iteration_idx": iteration_idx,
                "accepted": accepted,
                "rule": candidate.to_dict(),
                "coverage_set": coverage_set,
                "marginal_coverage": marginal,
                "remaining_size": len(uncovered_doc_ids),
                "eval_rows": eval_rows,
            }
        )

        if not accepted and marginal < min_coverage:
            termination_reason = "min_coverage_threshold"
            break
        if not accepted and marginal < min_marginal:
            termination_reason = "min_marginal_coverage"
            break
        if non_progress_count >= patience:
            termination_reason = "patience"
            break
    else:
        if not uncovered_doc_ids:
            termination_reason = "full_coverage"
        elif len(frozen_rules) >= max_rules:
            termination_reason = "max_rules"

    _write_seq_cover_trajectory(
        logger,
        {
            "query_idx": query_idx,
            "iterations": iterations,
            "termination_reason": termination_reason,
            "remaining_doc_ids": uncovered_doc_ids,
            "frozen_rule_count": len(frozen_rules),
        },
    )
    return AgentResult(
        rules=frozen_rules,
        trajectory=iterations,
        total_cost_usd=total_cost,
        turns_used=total_turns,
        termination_reason=termination_reason,
    )


__all__ = ["run_seq_cover_agent_on_query"]
