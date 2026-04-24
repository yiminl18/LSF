"""Curriculum-mode tool-agent: turn-aware broad-to-narrow rule generation.

Relationship to diverse_core:
- Same action schema: {tool, generate}
- Same append-only rule pool + diversity signature gate
- Differs only in turn-phase scheduling + late-phase post-generate auto-apply

Three phases, roughly 1/3 of max_turns each:
- early:  pure generate; prompt encourages broad coverage
- mid:    soft nudge; prompt suggests calling batch_apply_rule before generating
- late:   after each accepted generate, the system automatically calls batch_apply_rule
          and appends coverage + unmatched_doc_ids to the observation so the agent
          can refine its next rule (batch_apply_rule is zero-cost regex, no LLM budget)

Rationale: in diverse mode, late-turn blind generation dominates (measured: 12% validation
/ 88% blind generate, $0.32/query Phase A), while trivial submit fires too early
(5.4 turns average, q3 BestAcc drops to 2%). Curriculum makes the generate-validate
rhythm explicit along the turn axis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Tolerates the occasional LLM duplicate-JSON output, same as core.py.
_DECODER = json.JSONDecoder()

from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.rules.range_rule_json import RangeRule
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
    _rule_diversity_signature,
)
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost


_CURRICULUM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "tool_agent"
    / "tool_agent_system_curriculum.txt"
)


_PHASE_HINTS: dict[str, str] = {
    "early": (
        "[PHASE: EARLY — BROAD EXPLORATION]\n"
        "Prioritize DIVERSITY of rules across different modes / anchors / window sizes. "
        "Generate rules directly without requiring prior validation. Aim for coverage breadth."
    ),
    "mid": (
        "[PHASE: MID — VALIDATION-AWARE]\n"
        "You have accumulated some rules. Now prefer to call `batch_apply_rule` "
        "before generating to confirm anchor correctness and narrow the window. "
        "Continue to add rules targeting unmatched docs."
    ),
    "late": (
        "[PHASE: LATE — AUTO-VALIDATED NARROWING]\n"
        "The system will AUTOMATICALLY run `batch_apply_rule` on each rule you generate "
        "and append coverage + unmatched_doc_ids to your observation. Use this feedback "
        "to refine your next rule: target the `unmatched_doc_ids` with complementary "
        "narrow rules, or fix a rule that showed low coverage."
    ),
}


def _load_curriculum_template() -> str:
    return _CURRICULUM_PROMPT_PATH.read_text(encoding="utf-8")


def _phase_for_turn(turn_idx: int, max_turns: int) -> str:
    """Split max_turns into thirds; boundaries floored to guarantee at least 1 turn each."""
    early_end = max(1, max_turns // 3)
    mid_end = max(early_end + 1, 2 * max_turns // 3)
    if turn_idx < early_end:
        return "early"
    if turn_idx < mid_end:
        return "mid"
    return "late"


def _build_curriculum_system_prompt(
    query_text: str,
    corpus_toc: str,
    generated_summary: str,
    phase_hint: str,
) -> str:
    template = _load_curriculum_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{generated_summary}", generated_summary or "(none yet)")
        .replace("{phase_hint}", phase_hint)
    )


def _format_auto_apply_feedback(data: dict[str, Any]) -> str:
    """Format batch_apply_rule per-doc results into a short observation appended after generate_accepted."""
    parts = [
        f"[auto-apply on corpus] coverage={data.get('coverage', 0):.2f}, "
        f"matched={data.get('matched_count', 0)}/{data.get('total_docs', 0)}"
    ]
    unmatched = data.get("unmatched_doc_ids") or []
    if unmatched:
        # Show at most 5 unmatched to keep observation compact
        shown = unmatched[:5]
        suffix = f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
        parts.append(f"unmatched_docs={shown}{suffix}")
    # Include span_preview for the first 3 matched docs so agent can see concrete hits
    previews: list[str] = []
    for p in (data.get("per_doc") or [])[:6]:
        if p.get("matched") and p.get("span_preview"):
            previews.append(f"  {p['doc_id']}: {p['span_preview'][:120]}")
            if len(previews) >= 3:
                break
    if previews:
        parts.append("sample matches:\n" + "\n".join(previews))
    return "\n".join(parts)


def run_curriculum_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Curriculum-mode agent: turn-phase aware generation + late-phase auto-apply.

    Returns AgentResult containing all generated rules (deduplicated by diversity signature).
    Termination reasons: "max_turns" (normal), "budget", "context_limit", "no_docs".
    """
    if not doc_contexts:
        return AgentResult(
            rules=[], trajectory=[], total_cost_usd=0.0, turns_used=0,
            termination_reason="no_docs",
        )

    corpus_toc = _build_corpus_toc(doc_contexts)
    action_schema = _build_diverse_action_schema()

    history: list[_ConversationTurn] = []
    total_cost = 0.0
    discovered_rules: list[RangeRule] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    registry = ToolRegistry(doc_contexts[0], peer_docs=doc_contexts[1:])

    max_turn_limit = max_turns if max_turns is not None else agent_config.max_turns_per_query
    corpus_id_label = f"corpus[{len(doc_contexts)}](curriculum)"
    phase_counts: dict[str, int] = {"early": 0, "mid": 0, "late": 0}

    for turn_index in range(max_turn_limit):
        if total_cost >= agent_config.budget_usd:
            return AgentResult(
                rules=discovered_rules, trajectory=[],
                total_cost_usd=total_cost, turns_used=turn_index,
                termination_reason="budget",
            )

        phase = _phase_for_turn(turn_index, max_turn_limit)
        phase_counts[phase] += 1
        phase_hint = _PHASE_HINTS[phase]

        compact_hist = _compact_history(history, keep_recent=agent_config.history_compaction_threshold)
        stale_warning = _detect_stale_anchor(history, agent_config.stale_anchor_threshold)
        generated_summary = _format_generated_summary(discovered_rules)
        system_prompt = _build_curriculum_system_prompt(
            query_text, corpus_toc, generated_summary, phase_hint
        )
        current_system = system_prompt + (stale_warning or "")
        prompt = _serialize_prompt(current_system, compact_hist)

        try:
            _ensure_request_within_model_context(
                prompt_text=prompt,
                max_output_tokens=agent_config.agent_max_tokens,
                llm_provider=agent_config.agent_llm_provider,
                llm_model=agent_config.agent_llm_model,
                stage_label=f"curriculum_agent q{query_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            print(f"    [turn {turn_index}] curriculum prompt over context limit: {exc}")
            return AgentResult(
                rules=discovered_rules, trajectory=[],
                total_cost_usd=total_cost, turns_used=turn_index,
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
            print(f"    [turn {turn_index}] curriculum LLM call failed: {exc}")
            break

        agent_latency = (time.time() - t0) * 1000
        agent_call_cost = compute_cost(
            cache_result.input_tokens, cache_result.output_tokens,
            agent_config.agent_llm_provider,
            model=agent_config.agent_llm_model,
        )
        total_cost += agent_call_cost

        try:
            action, _end = _DECODER.raw_decode(cache_result.response.lstrip())
        except json.JSONDecodeError:
            history.append(_ConversationTurn(
                turn_index=turn_index, action_json=cache_result.response[:500],
                tool_name=None, observation="ERROR: invalid JSON response",
            ))
            logger.log_turn(
                query_idx=query_idx, doc_id=corpus_id_label,
                turn_index=turn_index, agent_reasoning="",
                tool_name="invalid_json", tool_args={},
                tool_result_preview="ERROR: invalid JSON response",
                cost_usd=agent_call_cost, latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=path_idx,
                tool_result_full={"error": "invalid_json", "phase": phase},
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
                history.append(_ConversationTurn(
                    turn_index=turn_index,
                    action_json=json.dumps(action, ensure_ascii=False),
                    tool_name="generate", observation=obs,
                ))
                logger.log_turn(
                    query_idx=query_idx, doc_id=corpus_id_label,
                    turn_index=turn_index, agent_reasoning=reasoning,
                    tool_name="generate_invalid", tool_args={"phase": phase},
                    tool_result_preview=obs[:200],
                    cost_usd=agent_call_cost, latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"error": "invalid_rule_payload",
                                      "phase": phase, "raw_rule": raw_rule},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            sig = _rule_diversity_signature(new_rule)
            if sig in seen_signatures:
                obs = (
                    f"REJECTED: rule duplicates an already-generated one on diversity "
                    f"signature {sig!r}. Vary mode / max_chars / anchor (first 50 chars). "
                    f"Already generated: {sorted(seen_signatures)!r}"
                )
                history.append(_ConversationTurn(
                    turn_index=turn_index,
                    action_json=json.dumps(action, ensure_ascii=False),
                    tool_name="generate", observation=obs,
                ))
                logger.log_turn(
                    query_idx=query_idx, doc_id=corpus_id_label,
                    turn_index=turn_index, agent_reasoning=reasoning,
                    tool_name="generate_rejected_duplicate",
                    tool_args={"signature": list(sig), "phase": phase},
                    tool_result_preview=obs[:200],
                    cost_usd=agent_call_cost, latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"duplicate_signature": list(sig), "phase": phase},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            # Append-only commit
            discovered_rules.append(new_rule)
            seen_signatures.add(sig)
            obs = f"Rule R{len(discovered_rules) - 1} accepted. Total: {len(discovered_rules)}."

            # ---- Late phase: post-generate auto-apply (zero LLM cost, regex only) ----
            auto_apply_data: dict[str, Any] | None = None
            if phase == "late":
                auto_result = registry.dispatch(
                    "batch_apply_rule",
                    {"rule_spec": new_rule.retrieval_spec.to_dict()},
                )
                if auto_result.success and isinstance(auto_result.data, dict):
                    auto_apply_data = auto_result.data
                    obs += "\n" + _format_auto_apply_feedback(auto_apply_data)
                # Note: auto_result.cost_usd is 0 for regex-only batch_apply_rule

            history.append(_ConversationTurn(
                turn_index=turn_index,
                action_json=json.dumps(action, ensure_ascii=False),
                tool_name="generate", observation=obs,
            ))
            logger.log_turn(
                query_idx=query_idx, doc_id=corpus_id_label,
                turn_index=turn_index, agent_reasoning=reasoning,
                tool_name="generate_accepted",
                tool_args={"rule_index": len(discovered_rules) - 1,
                           "signature": list(sig), "phase": phase},
                tool_result_preview=obs[:400],
                cost_usd=agent_call_cost, latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=path_idx,
                tool_result_full={
                    "rule_index": len(discovered_rules) - 1,
                    "rule": new_rule.to_dict(),
                    "phase": phase,
                    "auto_apply": auto_apply_data,
                },
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

        # action="tool" — inspection tools (batch_apply_rule / find_section / get_section / get_page / apply_rule)
        tool_name = action.get("tool", "")
        raw_args = action.get("args", "{}")
        try:
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            tool_args = {}

        tool_result = registry.dispatch(tool_name, tool_args)
        total_cost += tool_result.cost_usd

        if tool_result.success:
            obs_text = json.dumps(tool_result.data, ensure_ascii=False, default=str)
        else:
            obs_text = f"ERROR: {tool_result.error}"
        obs_text = _truncate_observation(obs_text, agent_config.observation_max_chars)

        history.append(_ConversationTurn(
            turn_index=turn_index,
            action_json=json.dumps(action, ensure_ascii=False),
            tool_name=tool_name, observation=obs_text,
        ))
        logger.log_turn(
            query_idx=query_idx, doc_id=corpus_id_label,
            turn_index=turn_index, agent_reasoning=reasoning,
            tool_name=tool_name, tool_args={**tool_args, "phase": phase},
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
        print(f"    [curriculum {phase} t{turn_index}] {tool_name} → {obs_text[:80]}")

    # max_turns reached: auto-commit accumulated rules
    return AgentResult(
        rules=discovered_rules, trajectory=[],
        total_cost_usd=total_cost, turns_used=max_turn_limit,
        termination_reason="max_turns",
    )


__all__ = [
    "_phase_for_turn",
    "_format_auto_apply_feedback",
    "run_curriculum_agent_on_query",
]
