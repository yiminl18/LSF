"""Reflexion-mode tool-agent: generate, evaluate failures, then refine."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace as dc_replace
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
from agent.tool_agent.diverse_core import (
    _build_diverse_action_schema,
    _format_generated_summary,
    _parse_rule_payload,
)
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.cache import CachedLLMCaller

_DECODER = json.JSONDecoder()

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "tool_agent"
_REFLEXION_PROMPT_PATH = _PROMPT_DIR / "tool_agent_system_reflexion.txt"
_FAILURE_BLOCK_PATH = _PROMPT_DIR / "tool_agent_reflexion_failure_block_v1.txt"


def _load_reflexion_template() -> str:
    return _REFLEXION_PROMPT_PATH.read_text(encoding="utf-8")


def _load_failure_template() -> str:
    return _FAILURE_BLOCK_PATH.read_text(encoding="utf-8")


def _rule_action_hash(rule: RangeRule) -> str:
    payload = json.dumps(rule.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_reflexion_system_prompt(
    query_text: str,
    corpus_toc: str,
    generated_summary: str,
    failure_memory: str,
) -> str:
    template = _load_reflexion_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{generated_summary}", generated_summary or "(none yet)")
        .replace("{failure_memory}", failure_memory or "(none yet)")
    )


def _serialize_rule_key(rule: RangeRule) -> str:
    return json.dumps(rule.retrieval_spec.to_dict(), ensure_ascii=False, sort_keys=True)


def _dedupe_rules(rules: list[RangeRule]) -> list[RangeRule]:
    seen: set[str] = set()
    unique: list[RangeRule] = []
    for rule in rules:
        key = _serialize_rule_key(rule)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def _run_reflexion_round(
    *,
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    round_idx: int,
    generated_rules: list[RangeRule],
    failure_memory: str,
    seen_action_hashes: set[str],
    force_distinct_action: bool,
) -> AgentResult:
    corpus_toc = _build_corpus_toc(doc_contexts)
    action_schema = _build_diverse_action_schema()
    history: list[_ConversationTurn] = []
    discovered_rules: list[RangeRule] = []
    total_cost = 0.0
    registry = ToolRegistry(doc_contexts[0], peer_docs=doc_contexts[1:])
    corpus_id_label = f"corpus[{len(doc_contexts)}](reflexion:r{round_idx})"
    nav_turn_counter = 0

    for turn_index in range(agent_config.max_turns_per_query):
        if total_cost >= agent_config.budget_usd:
            return AgentResult(
                rules=discovered_rules,
                trajectory=[],
                total_cost_usd=total_cost,
                turns_used=turn_index,
                termination_reason="budget",
            )

        compact_hist = _compact_history(
            history,
            keep_recent=agent_config.history_compaction_threshold,
        )
        generated_summary = _format_generated_summary([*generated_rules, *discovered_rules])
        system_prompt = _build_reflexion_system_prompt(
            query_text=query_text,
            corpus_toc=corpus_toc,
            generated_summary=generated_summary,
            failure_memory=failure_memory,
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
                stage_label=f"reflexion_agent q{query_idx} round={round_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            print(f"    [round {round_idx} turn {turn_index}] reflexion prompt over context limit: {exc}")
            return AgentResult(
                rules=discovered_rules,
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
            print(f"    [round {round_idx} turn {turn_index}] reflexion LLM call failed: {exc}")
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
                path_idx=round_idx,
                tool_result_full={"error": "invalid_json", "round_idx": round_idx},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

        action_type = action.get("action", "")
        reasoning = action.get("reasoning", "")

        if action_type == "generate":
            raw_rule = action.get("rule")
            new_rule = _parse_rule_payload(raw_rule)
            if new_rule is None:
                obs = "ERROR: 'rule' missing or malformed; provide a valid retrieval_spec."
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
                    tool_args={},
                    tool_result_preview=obs,
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=round_idx,
                    tool_result_full={"error": "invalid_rule_payload", "raw_rule": raw_rule},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            action_hash = _rule_action_hash(new_rule)
            if force_distinct_action and action_hash in seen_action_hashes:
                obs = (
                    "REJECTED: this generated RangeRule is identical to a prior "
                    "reflexion action. Generate a different rule family, anchor, "
                    "direction, or window size."
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
                    tool_args={"action_hash": action_hash},
                    tool_result_preview=obs,
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=round_idx,
                    tool_result_full={"duplicate_action_hash": action_hash},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            seen_action_hashes.add(action_hash)
            discovered_rules.append(new_rule)
            obs = (
                f"Rule R{len(generated_rules) + len(discovered_rules) - 1} "
                f"accepted for reflexion round {round_idx}."
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
                tool_name="generate_accepted",
                tool_args={"round_idx": round_idx, "action_hash": action_hash},
                tool_result_preview=obs,
                cost_usd=agent_call_cost,
                latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=round_idx,
                tool_result_full={"rule": new_rule.to_dict(), "round_idx": round_idx},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

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
            path_idx=round_idx,
            tool_result_full=tool_result.data if tool_result.success else {"error": tool_result.error},
            prompt_text=prompt,
            raw_response=cache_result.response,
        )

    return AgentResult(
        rules=discovered_rules,
        trajectory=[],
        total_cost_usd=total_cost,
        turns_used=agent_config.max_turns_per_query,
        termination_reason="max_turns",
    )


def _evaluate_rule_pool(
    rules: list[RangeRule],
    doc_contexts: list[DocumentContext],
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    remaining_budget: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    evaluations: list[dict[str, Any]] = []
    total_cost = 0.0
    doc_count = len(doc_contexts)

    for rule_idx, rule in enumerate(rules):
        failures: list[dict[str, Any]] = []
        success_doc_ids: list[str] = []
        matched_doc_ids: list[str] = []
        matched_count = 0
        success_count = 0
        for doc in doc_contexts:
            if remaining_budget is not None and total_cost >= remaining_budget:
                break
            subset = execute_range_rule(rule, doc.normalized_text)
            span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
            if not subset.matched or not span_text.strip():
                failures.append(
                    {
                        "doc_id": doc.doc_id,
                        "rule_index": rule_idx,
                        "blocker": subset.metadata.get("reason", "empty_span"),
                        "predicted_span": "",
                        "gold_answer": doc.ground_truth,
                    }
                )
                continue

            matched_count += 1
            matched_doc_ids.append(doc.doc_id)
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
            if score.judge_result:
                success_count += 1
                success_doc_ids.append(doc.doc_id)
            else:
                failures.append(
                    {
                        "doc_id": doc.doc_id,
                        "rule_index": rule_idx,
                        "blocker": "judge_false",
                        "predicted_span": span_text[:500],
                        "generated_answer": score.generated_answer,
                        "gold_answer": doc.ground_truth,
                    }
                )

        coverage = matched_count / doc_count if doc_count else 0.0
        accuracy = success_count / doc_count if doc_count else 0.0
        evaluations.append(
            {
                "rule_index": rule_idx,
                "rule_text": rule.rule_text,
                "retrieval_spec": rule.retrieval_spec.to_dict(),
                "coverage": round(coverage, 4),
                "accuracy": round(accuracy, 4),
                "score": round(coverage * accuracy, 4),
                "matched_count": matched_count,
                "success_count": success_count,
                "total_docs": doc_count,
                "success_doc_ids": success_doc_ids,
                "matched_doc_ids": matched_doc_ids,
                "failures": sorted(failures, key=lambda item: (item["rule_index"], item["doc_id"])),
            }
        )

    evaluations.sort(key=lambda item: (-item["accuracy"], -item["coverage"], item["rule_index"]))
    return evaluations, total_cost


def _render_failure_memory(best_evaluation: dict[str, Any] | None, failure_window: int) -> str:
    if best_evaluation is None:
        return ""
    failures = list(best_evaluation.get("failures", []))[:failure_window]
    payload = {
        "current_best_rule": {
            "rule_index": best_evaluation.get("rule_index"),
            "rule_text": best_evaluation.get("rule_text"),
            "accuracy": best_evaluation.get("accuracy"),
            "coverage": best_evaluation.get("coverage"),
        },
        "representative_failures": failures,
    }
    rendered_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return _load_failure_template().replace("{failure_memory_json}", rendered_json)


def _write_reflexion_trajectory(
    logger: TrajectoryLogger,
    payload: dict[str, Any],
) -> None:
    output_dir = getattr(logger, "_path").parent
    (output_dir / "reflexion_trajectory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_reflexion_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Run reflexion rounds and return the best RangeRule pool."""
    if not doc_contexts:
        return AgentResult(
            rules=[],
            trajectory=[],
            total_cost_usd=0.0,
            turns_used=0,
            termination_reason="no_docs",
        )

    max_iterations = int(getattr(agent_config, "reflexion_max_iterations", 3))
    failure_window = int(getattr(agent_config, "reflexion_failure_window", 4))
    force_distinct_action = bool(
        getattr(agent_config, "reflexion_force_distinct_action", True)
    )
    keep_best_only = bool(getattr(agent_config, "reflexion_keep_best_only", True))
    stop_on_no_improvement = bool(
        getattr(agent_config, "reflexion_stop_on_no_improvement", False)
    )
    budget_factor = float(getattr(agent_config, "reflexion_per_round_budget_factor", 1.0))
    configured_round_budget = max(0.0, agent_config.budget_usd * budget_factor)

    round_turns = (
        max_turns
        if max_turns is not None
        else int(getattr(agent_config, "reflexion_per_round_max_turns", agent_config.max_turns_per_query))
    )

    rule_pool: list[RangeRule] = []
    seen_action_hashes: set[str] = set()
    rounds: list[dict[str, Any]] = []
    total_cost = 0.0
    total_turns = 0
    best_evaluation: dict[str, Any] | None = None
    previous_best_accuracy: float | None = None
    termination_reason = "max_iterations"

    for round_idx in range(max_iterations):
        remaining_budget = agent_config.budget_usd - total_cost
        if remaining_budget <= 0:
            termination_reason = "budget"
            break
        round_config = dc_replace(
            agent_config,
            max_turns_per_query=max(1, round_turns),
            budget_usd=min(configured_round_budget, remaining_budget),
        )
        failure_memory = (
            _render_failure_memory(best_evaluation, failure_window)
            if round_idx > 0
            else ""
        )
        round_result = _run_reflexion_round(
            query_text=query_text,
            query_idx=query_idx,
            doc_contexts=doc_contexts,
            cached_caller=cached_caller,
            agent_config=round_config,
            logger=logger,
            round_idx=round_idx,
            generated_rules=rule_pool,
            failure_memory=failure_memory,
            seen_action_hashes=seen_action_hashes,
            force_distinct_action=force_distinct_action,
        )
        total_cost += round_result.total_cost_usd
        total_turns += round_result.turns_used
        rule_pool = _dedupe_rules([*rule_pool, *round_result.rules])

        remaining_budget = agent_config.budget_usd - total_cost
        if remaining_budget <= 0:
            rounds.append(
                {
                    "round_idx": round_idx,
                    "new_rules": len(round_result.rules),
                    "pool_size": len(rule_pool),
                    "best_accuracy": (
                        float(best_evaluation["accuracy"])
                        if best_evaluation is not None
                        else 0.0
                    ),
                    "best_rule_index": (
                        best_evaluation.get("rule_index")
                        if best_evaluation is not None
                        else None
                    ),
                    "failure_count": (
                        len(best_evaluation.get("failures", []))
                        if best_evaluation is not None
                        else 0
                    ),
                    "termination_reason": round_result.termination_reason,
                }
            )
            termination_reason = "budget"
            break

        evaluations, eval_cost = _evaluate_rule_pool(
            rule_pool,
            doc_contexts,
            query_text,
            cached_caller,
            agent_config.agent_llm_provider,
            agent_config.agent_llm_model,
            remaining_budget=remaining_budget,
        )
        total_cost += eval_cost
        best_evaluation = evaluations[0] if evaluations else None
        current_best_accuracy = (
            float(best_evaluation["accuracy"]) if best_evaluation is not None else 0.0
        )

        if keep_best_only and best_evaluation is not None and rule_pool:
            rule_pool = [rule_pool[int(best_evaluation["rule_index"])]]
            best_evaluation = {**best_evaluation, "rule_index": 0}

        rounds.append(
            {
                "round_idx": round_idx,
                "new_rules": len(round_result.rules),
                "pool_size": len(rule_pool),
                "best_accuracy": current_best_accuracy,
                "best_rule_index": (
                    best_evaluation.get("rule_index") if best_evaluation is not None else None
                ),
                "failure_count": (
                    len(best_evaluation.get("failures", []))
                    if best_evaluation is not None
                    else 0
                ),
                "termination_reason": round_result.termination_reason,
            }
        )

        if stop_on_no_improvement and previous_best_accuracy is not None:
            if current_best_accuracy <= previous_best_accuracy:
                termination_reason = "no_improvement"
                break
        if total_cost >= agent_config.budget_usd:
            termination_reason = "budget"
            break
        previous_best_accuracy = current_best_accuracy
    else:
        if best_evaluation is not None and float(best_evaluation.get("accuracy", 0.0)) >= 1.0:
            termination_reason = "max_iterations"

    _write_reflexion_trajectory(
        logger,
        {
            "query_idx": query_idx,
            "rounds": rounds,
            "termination_reason": termination_reason,
            "best_rule": rule_pool[0].to_dict() if rule_pool else None,
        },
    )
    return AgentResult(
        rules=rule_pool,
        trajectory=rounds,
        total_cost_usd=total_cost,
        turns_used=total_turns,
        termination_reason=termination_reason,
    )


__all__ = [
    "_render_failure_memory",
    "run_reflexion_agent_on_query",
]
