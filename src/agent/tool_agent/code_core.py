"""Code-mode tool-agent: discover sandboxed CodeRules."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from agent.prompts.tool_agent.tool_agent_code_action_schema import (
    _build_code_action_schema,
)
from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.rules.code_rule_json import CodeRule
from agent.rules.code_rule_sandbox import execute_locate_region, validate_code_ast
from agent.rules.range_rule_scorer import score_retrieved_subset
from agent.tool_agent.core import (
    AgentConfig,
    AgentResult,
    _build_corpus_toc,
    _compact_history,
    _ConversationTurn,
    _serialize_prompt,
    _truncate_observation,
)
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.cache import CachedLLMCaller

_DECODER = json.JSONDecoder()
_LOG = logging.getLogger(__name__)
_LOCATE_REGION_DEF_RE = re.compile(r"^\s*def\s+locate_region\s*\(", re.MULTILINE)
_MAX_BEST_RULES = 5

_CODE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "tool_agent"
    / "tool_agent_system_code.txt"
)


def _load_code_template() -> str:
    return _CODE_PROMPT_PATH.read_text(encoding="utf-8")


def _format_generated_summary(rules: list[CodeRule]) -> str:
    if not rules:
        return ""
    lines: list[str] = []
    for idx, rule in enumerate(rules):
        first_line = rule.code.strip().splitlines()[0] if rule.code.strip() else ""
        lines.append(f"  C{idx}: {rule.rule_text[:80]} code={first_line!r}")
    return "\n".join(lines)


def _build_code_system_prompt(
    query_text: str,
    corpus_toc: str,
    generated_summary: str,
) -> str:
    template = _load_code_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{generated_summary}", generated_summary or "(none yet)")
    )


def _parse_code_rule_payload(
    raw_rule: Any,
) -> tuple[CodeRule | None, list[str], str | None]:
    if isinstance(raw_rule, str):
        try:
            raw_rule = json.loads(raw_rule)
        except json.JSONDecodeError as exc:
            return None, [], f"invalid JSON rule payload: {exc}"
    if not isinstance(raw_rule, dict):
        return None, [], "rule must be a JSON object"
    code = raw_rule.get("code")
    if not isinstance(code, str) or not code.strip():
        return None, [], "rule.code must be a non-empty string"

    violations = []
    if not _LOCATE_REGION_DEF_RE.search(code):
        violations.append("missing locate_region function")
    violations.extend(validate_code_ast(code))
    if violations:
        return None, violations, None

    return (
        CodeRule(
            rule_text=str(raw_rule.get("rule_text", "code_rule")),
            evidence_basis=str(raw_rule.get("evidence_basis", "code_generated")),
            code=code.strip(),
        ),
        [],
        None,
    )


def _code_rule_key(rule: CodeRule) -> str:
    return json.dumps({"rule_kind": "code", "code": rule.code}, sort_keys=True)


def _dedupe_code_rules(rules: list[CodeRule]) -> list[CodeRule]:
    seen: set[str] = set()
    unique: list[CodeRule] = []
    for rule in rules:
        key = _code_rule_key(rule)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def _code_rule_entry(rule: CodeRule) -> dict[str, Any]:
    return {
        "rule_kind": "code",
        "rule_text": rule.rule_text,
        "evidence_basis": rule.evidence_basis,
        "code": rule.code,
        "sandbox_validation_status": "valid",
    }


def _rejection_counts_from_agent_result(result: AgentResult) -> dict[str, int]:
    if not result.trajectory:
        return {"ast": 0, "timeout": 0, "runtime_error": 0}
    counts = result.trajectory[0].get("code_rule_rejections", {})
    return {
        "ast": int(counts.get("ast", 0)),
        "timeout": int(counts.get("timeout", 0)),
        "runtime_error": int(counts.get("runtime_error", 0)),
    }


def _rejection_counts_from_eval(cross_doc_eval: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "ast": sum(int(row.get("sandbox_ast_count", 0)) for row in cross_doc_eval),
        "timeout": sum(int(row.get("sandbox_timeout_count", 0)) for row in cross_doc_eval),
        "runtime_error": sum(
            int(row.get("sandbox_runtime_error_count", 0)) for row in cross_doc_eval
        ),
    }


def _merge_rejection_counts(
    agent_counts: dict[str, int],
    eval_counts: dict[str, int],
) -> dict[str, int]:
    return {
        "ast": int(agent_counts.get("ast", 0)) + int(eval_counts.get("ast", 0)),
        "timeout": int(agent_counts.get("timeout", 0))
        + int(eval_counts.get("timeout", 0)),
        "runtime_error": int(agent_counts.get("runtime_error", 0))
        + int(eval_counts.get("runtime_error", 0)),
    }


def _classify_sandbox_error(error: str | None) -> str | None:
    if not error:
        return None
    if error.startswith("AST violations:"):
        return "ast"
    if error.startswith("TimeoutError:"):
        return "timeout"
    return "runtime_error"


def _cross_doc_evaluate_code(
    rules: list[CodeRule],
    doc_contexts: list[DocumentContext],
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    remaining_budget: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cum_eval_cost = 0.0
    budget_exhausted = False

    for rule_index, rule in enumerate(rules):
        if remaining_budget is not None and cum_eval_cost >= remaining_budget:
            if not budget_exhausted:
                _LOG.info("code cross-doc eval budget exhausted at rule %s", rule_index)
                budget_exhausted = True
            rows.append(
                {
                    "rule_index": rule_index,
                    "rule_kind": "code",
                    "rule_text": rule.rule_text,
                    "code": rule.code,
                    "coverage": 0.0,
                    "accuracy": 0.0,
                    "score": 0.0,
                    "matched_count": 0,
                    "success_count": 0,
                    "evaluated_docs": 0,
                    "total_docs": len(doc_contexts),
                    "eval_cost_usd": 0.0,
                    "skipped_due_to_budget": True,
                    "success_doc_ids": [],
                    "matched_doc_ids": [],
                    "sandbox_validation_status": "valid",
                }
            )
            continue

        matched_count = 0
        success_count = 0
        evaluated_doc_count = 0
        total_cost = 0.0
        success_doc_ids: list[str] = []
        matched_doc_ids: list[str] = []
        sandbox_counts = {"ast": 0, "timeout": 0, "runtime_error": 0}

        for doc in doc_contexts:
            if remaining_budget is not None and cum_eval_cost >= remaining_budget:
                break
            evaluated_doc_count += 1
            exec_result = execute_locate_region(rule.code, doc.normalized_text)
            if not exec_result.success:
                reason = _classify_sandbox_error(exec_result.error)
                if reason is not None:
                    sandbox_counts[reason] += 1
                continue

            region_text = exec_result.returned_region
            if not region_text.strip():
                continue
            matched_count += 1
            matched_doc_ids.append(doc.doc_id)
            score = score_retrieved_subset(
                question=query_text,
                retrieved_text=region_text,
                ground_truth=doc.ground_truth,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            cost = (
                score.metadata["generation"]["cost_usd"]
                + score.metadata["judge"]["cost_usd"]
            )
            total_cost += cost
            cum_eval_cost += cost
            if score.judge_result:
                success_count += 1
                success_doc_ids.append(doc.doc_id)

        doc_count = len(doc_contexts)
        coverage = matched_count / doc_count if doc_count else 0.0
        accuracy = success_count / doc_count if doc_count else 0.0
        rows.append(
            {
                "rule_index": rule_index,
                "rule_kind": "code",
                "rule_text": rule.rule_text,
                "code": rule.code,
                "coverage": round(coverage, 4),
                "accuracy": round(accuracy, 4),
                "score": round(coverage * accuracy, 4),
                "matched_count": matched_count,
                "success_count": success_count,
                "evaluated_docs": evaluated_doc_count,
                "total_docs": doc_count,
                "budget_truncated": evaluated_doc_count < doc_count,
                "eval_cost_usd": round(total_cost, 6),
                "skipped_due_to_budget": False,
                "success_doc_ids": success_doc_ids,
                "matched_doc_ids": matched_doc_ids,
                "sandbox_validation_status": "valid",
                "sandbox_ast_count": sandbox_counts["ast"],
                "sandbox_timeout_count": sandbox_counts["timeout"],
                "sandbox_runtime_error_count": sandbox_counts["runtime_error"],
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def run_code_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Code-mode agent: append-only CodeRule generation."""
    if not doc_contexts:
        return AgentResult(
            rules=[],
            trajectory=[{"code_rule_rejections": {"ast": 0, "timeout": 0, "runtime_error": 0}}],
            total_cost_usd=0.0,
            turns_used=0,
            termination_reason="no_docs",
        )

    corpus_toc = _build_corpus_toc(doc_contexts)
    action_schema = _build_code_action_schema()
    history: list[_ConversationTurn] = []
    total_cost = 0.0
    discovered_rules: list[CodeRule] = []
    rejection_counts = {"ast": 0, "timeout": 0, "runtime_error": 0}
    registry = ToolRegistry(doc_contexts[0], peer_docs=doc_contexts[1:])
    max_turn_limit = max_turns if max_turns is not None else agent_config.max_turns_per_query
    corpus_id_label = f"corpus[{len(doc_contexts)}](code)"
    nav_turn_counter = 0

    for turn_index in range(max_turn_limit):
        if total_cost >= agent_config.budget_usd:
            return AgentResult(
                rules=discovered_rules,
                trajectory=[{"code_rule_rejections": rejection_counts}],
                total_cost_usd=total_cost,
                turns_used=turn_index,
                termination_reason="budget",
            )

        compact_hist = _compact_history(
            history,
            keep_recent=agent_config.history_compaction_threshold,
        )
        system_prompt = _build_code_system_prompt(
            query_text,
            corpus_toc,
            _format_generated_summary(discovered_rules),
        )
        if nav_turn_counter >= agent_config.nav_turn_cap:
            system_prompt += (
                f"\n\n[system] Navigation budget exhausted "
                f"({agent_config.nav_turn_cap} consecutive find_section calls). "
                "Next action MUST be `generate` or `try_code_rule`."
            )
        prompt = _serialize_prompt(system_prompt, compact_hist)

        try:
            _ensure_request_within_model_context(
                prompt_text=prompt,
                max_output_tokens=agent_config.agent_max_tokens,
                llm_provider=agent_config.agent_llm_provider,
                llm_model=agent_config.agent_llm_model,
                stage_label=f"code_agent q{query_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            _LOG.warning("code agent prompt over context limit: %s", exc)
            return AgentResult(
                rules=discovered_rules,
                trajectory=[{"code_rule_rejections": rejection_counts}],
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
            _LOG.warning("code agent LLM call failed: %s", exc)
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
                path_idx=path_idx,
                tool_result_full={"error": "invalid_json"},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

        action_type = action.get("action", "")
        reasoning = action.get("reasoning", "")
        if action_type == "generate":
            rule, violations, error = _parse_code_rule_payload(action.get("rule"))
            if violations:
                rejection_counts["ast"] += 1
                obs = f"REJECTED: code failed sandbox AST validation: {violations}"
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
                    tool_name="generate_rejected_ast",
                    tool_args={},
                    tool_result_preview=obs[:500],
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"violations": violations},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue
            if rule is None:
                obs = f"ERROR: invalid code rule payload: {error}"
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
                    tool_result_preview=obs[:500],
                    cost_usd=agent_call_cost,
                    latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"error": error, "raw_rule": action.get("rule")},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            discovered_rules.append(rule)
            obs = f"CodeRule C{len(discovered_rules) - 1} accepted."
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
                tool_args={"rule_index": len(discovered_rules) - 1},
                tool_result_preview=obs,
                cost_usd=agent_call_cost,
                latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=path_idx,
                tool_result_full={
                    "rule_index": len(discovered_rules) - 1,
                    "rule": _code_rule_entry(rule),
                },
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
        elif tool_name in {"batch_apply_rule", "try_code_rule"}:
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
            path_idx=path_idx,
            tool_result_full=tool_result.data if tool_result.success else {"error": tool_result.error},
            prompt_text=prompt,
            raw_response=cache_result.response,
        )

    return AgentResult(
        rules=discovered_rules,
        trajectory=[{"code_rule_rejections": rejection_counts}],
        total_cost_usd=total_cost,
        turns_used=max_turn_limit,
        termination_reason="max_turns",
    )


def save_code_phase_a_results(
    output_dir: Path,
    query_idx: int,
    best_rules: list[CodeRule],
    cross_doc_eval: list[dict[str, Any]],
    total_cost: float,
    processed_doc_ids: list[str],
    exploration_cost: float,
    cross_doc_eval_cost: float,
    excluded_doc_ids: list[str],
    code_rule_rejections: dict[str, int],
) -> None:
    rules_data = {
        "query_idx": query_idx,
        "packaging_mode": "tool_agent_phase_a",
        "rule_mode": "python_code",
        "merged_rules": [_code_rule_entry(rule) for rule in best_rules],
        "cross_doc_eval": cross_doc_eval,
        "code_rule_rejections": code_rule_rejections,
        "exploration_cost_usd": round(exploration_cost, 4),
        "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    }
    with (output_dir / "best_rules.json").open("w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)

    with (output_dir / "phase_a_docs.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "query_idx": query_idx,
                "excluded_doc_ids": excluded_doc_ids,
                "processed_doc_ids": processed_doc_ids,
                "count": len(excluded_doc_ids),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [f"# Phase A Report: q{query_idx}\n"]
    lines.append(f"**CodeRules discovered**: {len(best_rules)}")
    lines.append(
        f"**Costs**: exploration ${exploration_cost:.4f} + "
        f"cross_doc_eval ${cross_doc_eval_cost:.4f} = total ${total_cost:.4f}\n"
    )
    lines.append("## Code Cross-Doc Evaluation\n")
    lines.append("| Rule | Coverage | Accuracy | Score | Matched | Success |")
    lines.append("|------|----------|----------|-------|---------|---------|")
    for row in cross_doc_eval:
        lines.append(
            f"| C{row['rule_index']} | {row['coverage']:.1%} | "
            f"{row['accuracy']:.1%} | {row['score']:.3f} | "
            f"{row['matched_count']}/{row['total_docs']} | "
            f"{row['success_count']}/{row['total_docs']} |"
        )
    with (output_dir / "phase_a_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with (output_dir / "phase_a_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "query_idx": query_idx,
                "rule_count": len(best_rules),
                "cross_doc_eval": cross_doc_eval,
                "code_rule_rejections": code_rule_rejections,
                "exploration_cost_usd": round(exploration_cost, 4),
                "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
                "total_cost_usd": round(total_cost, 4),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


__all__ = [
    "_cross_doc_evaluate_code",
    "_dedupe_code_rules",
    "_merge_rejection_counts",
    "_rejection_counts_from_agent_result",
    "_rejection_counts_from_eval",
    "run_code_agent_on_query",
    "save_code_phase_a_results",
]
