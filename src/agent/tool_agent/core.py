"""Agent Core Loop — per-query rule validation loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Azure gpt-5.4-mini occasionally duplicates the same Structured-Output JSON at the
# end of a response. raw_decode takes only the first valid object and ignores the
# trailing copy — more robust than strict json.loads.
_DECODER = json.JSONDecoder()

from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.rules.range_rule_json import RangeRule, RetrievalSpec
from agent.tool_agent import toc
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost


@dataclass
class AgentConfig:
    """Agent runtime configuration."""

    max_turns_per_query: int = 15
    nav_turn_cap: int = 4
    budget_usd: float = 2.0
    stale_anchor_threshold: int = 3
    agent_llm_provider: str = "azure"
    agent_llm_model: str = "gpt-5.4-mini"
    agent_max_tokens: int = 1500
    observation_max_chars: int = 8000
    history_compaction_threshold: int = 12


@dataclass
class AgentResult:
    """Agent run result for a single query."""

    rules: list[RangeRule]
    trajectory: list[dict[str, Any]]
    total_cost_usd: float
    turns_used: int
    termination_reason: str  # "submit" | "budget" | "max_turns" | "no_docs" | "context_limit"


_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "tool_agent"
    / "tool_agent_system_v2.txt"
)

def _load_system_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_system_prompt(
    query_text: str,
    corpus_toc: str,
    tried_rules_summary: str,
) -> str:
    template = _load_system_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{tried_rules_summary}", tried_rules_summary or "None yet.")
    )


def _build_corpus_toc(doc_contexts: list[DocumentContext]) -> str:
    """Build a lightweight TOC view of sampled docs (replaces the old full-text bundle)."""
    return toc.build_toc(doc_contexts)


def _build_action_schema() -> dict[str, Any]:
    return {
        "name": "agent_action",
        "description": "Agent tool call or rule submission",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["tool", "submit"]},
                "reasoning": {"type": "string"},
                "tool": {"type": ["string", "null"]},
                "args": {"type": ["string", "null"]},
                "rules": {"type": "string"},
            },
            "required": ["action", "reasoning", "tool", "args", "rules"],
        },
    }


@dataclass
class _ConversationTurn:
    turn_index: int
    action_json: str
    tool_name: str | None
    observation: str


def _serialize_prompt(system_prompt: str, history: list[_ConversationTurn]) -> str:
    parts: list[str] = [f"<system>\n{system_prompt}\n</system>"]
    for turn in history:
        parts.append(
            f"\n<turn idx={turn.turn_index}>\n"
            f"Agent: {turn.action_json}\n"
            f"Observation: {turn.observation}\n"
            f"</turn>"
        )
    parts.append("\n\nBased on the observations above, decide your next action.")
    return "\n".join(parts)


def _compact_history(
    history: list[_ConversationTurn],
    keep_recent: int = 5,
) -> list[_ConversationTurn]:
    if len(history) <= keep_recent:
        return history

    compacted: list[_ConversationTurn] = []
    cutoff = len(history) - keep_recent
    for turn in history[:cutoff]:
        summary_obs = turn.observation[:120].replace("\n", " ")
        compacted.append(
            _ConversationTurn(
                turn_index=turn.turn_index,
                action_json=json.dumps(
                    {"action": "tool", "tool": turn.tool_name, "reasoning": "..."},
                    ensure_ascii=False,
                ),
                tool_name=turn.tool_name,
                observation=f"[summary] {summary_obs}...",
            )
        )
    compacted.extend(history[cutoff:])
    return compacted


def _truncate_observation(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total {len(text)} chars]"


def _detect_stale_anchor(
    history: list[_ConversationTurn],
    threshold: int = 3,
) -> str | None:
    recent_anchors: list[str] = []
    for turn in reversed(history):
        if turn.tool_name not in {"apply_rule", "batch_apply_rule"}:
            continue
        try:
            action = json.loads(turn.action_json)
            raw_args = action.get("args") or "{}"
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            anchor = (args_dict or {}).get("rule_spec", {}).get("anchor", "")
            if anchor:
                recent_anchors.append(anchor)
        except (json.JSONDecodeError, AttributeError):
            continue
        if len(recent_anchors) >= threshold:
            break

    if len(recent_anchors) >= threshold and len(set(recent_anchors)) == 1:
        return (
            f"\n\nWARNING: You have used the anchor \"{recent_anchors[0]}\" "
            f"{threshold} times without improving the rule. Try a different anchor, "
            "or switch between a value-format regex and a stable label-based after rule."
        )
    return None


def run_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    tried_rules: list[dict[str, Any]] | None = None,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Single per-query agent: reads sampled doc full text and validates rules with tools."""
    if not doc_contexts:
        return AgentResult(
            rules=[],
            trajectory=[],
            total_cost_usd=0.0,
            turns_used=0,
            termination_reason="no_docs",
        )

    tried_summary = ""
    if tried_rules:
        tried_summary = "\n".join(
            f"- {tr.get('rule_text', '')[:80]} → {tr.get('result', 'unknown')}"
            for tr in tried_rules
        )

    corpus_toc = _build_corpus_toc(doc_contexts)
    system_prompt = _build_system_prompt(query_text, corpus_toc, tried_summary)
    action_schema = _build_action_schema()

    history: list[_ConversationTurn] = []
    total_cost = 0.0
    discovered_rules: list[RangeRule] = []
    registry = ToolRegistry(doc_contexts[0], peer_docs=doc_contexts[1:])

    max_turn_limit = (
        max_turns
        if max_turns is not None
        else agent_config.max_turns_per_query
    )
    corpus_id_label = f"corpus[{len(doc_contexts)}]"
    nav_turn_counter: int = 0

    for turn_index in range(max_turn_limit):
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
        stale_warning = _detect_stale_anchor(history, agent_config.stale_anchor_threshold)
        current_system = system_prompt + (stale_warning or "")
        if nav_turn_counter >= agent_config.nav_turn_cap:
            current_system += (
                f"\n\n[system] Navigation budget exhausted "
                f"({agent_config.nav_turn_cap} consecutive find_section calls). "
                "You must now call batch_apply_rule or submit."
            )
        # Forced-submit last-chance phase: when ≤2 turns remain and no rules submitted yet
        turns_remaining = max_turn_limit - turn_index
        if turns_remaining <= 2 and not discovered_rules:
            current_system += (
                f"\n\n[system] LAST-CHANCE SUBMIT PHASE: you have {turns_remaining} "
                f"turn(s) left and still 0 rules submitted. Submit NOW using the best 3 rules "
                "from your batch_apply_rule history (choose rules with highest coverage). "
                "DO NOT call find_section or batch_apply_rule anymore — only respond with "
                "`action=\"submit\"` and fill `rules` with 3 retrieval_spec objects. "
                "If you do not submit this turn, your entire run loses ALL rules."
            )
        prompt = _serialize_prompt(current_system, compact_hist)

        try:
            _ensure_request_within_model_context(
                prompt_text=prompt,
                max_output_tokens=agent_config.agent_max_tokens,
                llm_provider=agent_config.agent_llm_provider,
                llm_model=agent_config.agent_llm_model,
                stage_label=f"tool_agent q{query_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            print(f"    [turn {turn_index}] Agent prompt exceeds context limit: {exc}")
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
            print(f"    [turn {turn_index}] Agent LLM call failed: {exc}")
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
            history.append(
                _ConversationTurn(
                    turn_index=turn_index,
                    action_json=cache_result.response[:500],
                    tool_name=None,
                    observation="ERROR: Invalid JSON response from agent.",
                )
            )
            logger.log_turn(
                query_idx=query_idx,
                doc_id=corpus_id_label,
                turn_index=turn_index,
                agent_reasoning="",
                tool_name="invalid_json",
                tool_args={},
                tool_result_preview="ERROR: Invalid JSON response from agent.",
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

        if action_type == "submit":
            raw_rules = action.get("rules", "[]")
            try:
                rules_data = json.loads(raw_rules) if isinstance(raw_rules, str) else raw_rules
            except json.JSONDecodeError:
                rules_data = []
            if not isinstance(rules_data, list):
                rules_data = []
            for rd in rules_data:
                if not isinstance(rd, dict):
                    print(f"    [turn {turn_index}] Skipping non-dict rule payload: {rd!r}")
                    continue
                retrieval_spec = rd.get("retrieval_spec")
                if not isinstance(retrieval_spec, dict):
                    print(f"    [turn {turn_index}] Skipping rule payload missing retrieval_spec")
                    continue
                try:
                    spec = RetrievalSpec(**retrieval_spec)
                    # 0-cost hint gate: agent may optionally attach answer_hint_pattern
                    # (already validated in batch_apply_rule); Phase B uses it as a
                    # zero-cost pre-filter. reliability is populated by _cross_doc_evaluate
                    # downstream (None here).
                    hint_pattern = rd.get("answer_hint_pattern")
                    if hint_pattern is not None and not isinstance(hint_pattern, str):
                        hint_pattern = None
                    discovered_rules.append(
                        RangeRule(
                            rule_text=rd.get("rule_text", "agent_rule"),
                            evidence_basis=rd.get("evidence_basis", "agent_discovered"),
                            retrieval_spec=spec,
                            answer_hint_pattern=hint_pattern,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    print(f"    [turn {turn_index}] Rule parse failed: {exc}")

            logger.log_turn(
                query_idx=query_idx,
                doc_id=corpus_id_label,
                turn_index=turn_index,
                agent_reasoning=reasoning,
                tool_name="submit",
                tool_args={"rule_count": len(rules_data)},
                tool_result_preview=f"Submitted {len(discovered_rules)} rules",
                cost_usd=agent_call_cost,
                latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=path_idx,
                tool_result_full={
                    "submitted_rule_count": len(discovered_rules),
                    "rules_data": rules_data,
                },
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            return AgentResult(
                rules=discovered_rules,
                trajectory=[],
                total_cost_usd=total_cost,
                turns_used=turn_index + 1,
                termination_reason="submit",
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

        action_json_str = json.dumps(action, ensure_ascii=False)
        history.append(
            _ConversationTurn(
                turn_index=turn_index,
                action_json=action_json_str,
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

        status = ""
        if tool_name == "batch_apply_rule" and tool_result.success:
            coverage = tool_result.data.get("coverage") if isinstance(tool_result.data, dict) else None
            if coverage is not None:
                status = f"cov={coverage}"
        print(f"    [turn {turn_index}] {tool_name} → {obs_text[:80]} {status}")

    return AgentResult(
        rules=discovered_rules,
        trajectory=[],
        total_cost_usd=total_cost,
        turns_used=max_turn_limit,
        termination_reason="max_turns",
    )
