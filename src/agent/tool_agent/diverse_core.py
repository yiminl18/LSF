"""Diverse-mode tool-agent: each turn appends ≥1 rule (no submit, system auto-commits).

Differences from `core.py:run_agent_on_query`:
- action enum: "tool" | "generate" (replaces "submit")
- each `generate` must include 1 rule that differs from already-generated rules on
  at least one of: mode, max_chars-bucket {0-200, 200-800, 800+}, anchor[:50] casefold.
- discovered_rules is append-only — agent CANNOT remove or amend prior rules.
- system auto-commits at max_turns: all accumulated rules go to cross_doc_eval + set-cover.
- batch_apply_rule / find_section / apply_rule still available as diagnostic tools.

Rationale: explore/exploit through forced diversity over time, instead of forcing the
agent to plan a complete set in one submit. Cross_doc_eval + set-cover (downstream)
filters the accumulated pool — this agent's job is to populate the pool with varied
candidates, not pick the winners.
"""

from __future__ import annotations

import json
import time
from typing import Any

# Tolerates the occasional LLM duplicate-JSON output, same as core.py.
_DECODER = json.JSONDecoder()

from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.rules.range_rule_json import RangeRule, RetrievalSpec
from agent.tool_agent.diversity import _max_chars_bucket, _rule_diversity_signature
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
from agent.tool_agent.document import DocumentContext
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.tools import ToolRegistry
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost


_DIVERSE_PROMPT_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "prompts"
    / "tool_agent"
    / "tool_agent_system_diverse.txt"
)


def _load_diverse_template() -> str:
    return _DIVERSE_PROMPT_PATH.read_text(encoding="utf-8")


def _build_diverse_system_prompt(
    query_text: str, corpus_toc: str, generated_summary: str
) -> str:
    template = _load_diverse_template()
    return (
        template.replace("{query_text}", query_text)
        .replace("{corpus_toc}", corpus_toc)
        .replace("{generated_summary}", generated_summary or "(none yet)")
    )


def _build_diverse_action_schema() -> dict[str, Any]:
    """action enum: 'tool' or 'generate' (no submit)."""
    return {
        "name": "agent_action",
        "description": "Agent tool call or single-rule generation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["tool", "generate"]},
                "reasoning": {"type": "string"},
                "tool": {"type": ["string", "null"]},
                "args": {"type": ["string", "null"]},
                "rule": {"type": ["string", "null"]},
            },
            "required": ["action", "reasoning", "tool", "args", "rule"],
        },
    }


def _format_generated_summary(rules: list[RangeRule]) -> str:
    if not rules:
        return ""
    lines: list[str] = []
    for i, r in enumerate(rules):
        spec = r.retrieval_spec
        lines.append(
            f"  R{i}: mode={spec.mode} max_chars={spec.max_chars} "
            f"anchor={(spec.anchor or '')[:60]!r}"
        )
    return "\n".join(lines)


_RETRIEVAL_SPEC_FIELDS: frozenset[str] = frozenset({
    "mode", "anchor", "anchor_b", "page_idx", "max_chars", "boundary_context_chars",
})


def _parse_rule_payload(rd: Any) -> RangeRule | None:
    """Parse a single rule payload from the agent action's 'rule' field.

    Defensively filters retrieval_spec to known fields — agents sometimes invent
    fields like `pattern` or `page_constraint` that don't exist on RetrievalSpec.
    Filtering preserves the rule instead of rejecting an otherwise valid candidate.
    """
    if isinstance(rd, str):
        try:
            rd = json.loads(rd)
        except json.JSONDecodeError:
            return None
    if not isinstance(rd, dict):
        return None
    retrieval_spec = rd.get("retrieval_spec")
    if not isinstance(retrieval_spec, dict):
        return None
    # Filter unknown fields silently (agent may invent kwargs)
    filtered_spec = {k: v for k, v in retrieval_spec.items() if k in _RETRIEVAL_SPEC_FIELDS}
    try:
        spec = RetrievalSpec(**filtered_spec)
    except (KeyError, TypeError, ValueError):
        return None
    hint_pattern = rd.get("answer_hint_pattern")
    if hint_pattern is not None and not isinstance(hint_pattern, str):
        hint_pattern = None
    return RangeRule(
        rule_text=rd.get("rule_text", "diverse_rule"),
        evidence_basis=rd.get("evidence_basis", "diverse_generated"),
        retrieval_spec=spec,
        answer_hint_pattern=hint_pattern,
    )


def run_diverse_agent_on_query(
    query_text: str,
    query_idx: int,
    doc_contexts: list[DocumentContext],
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    logger: TrajectoryLogger,
    max_turns: int | None = None,
    path_idx: int | None = None,
) -> AgentResult:
    """Diverse-mode agent: append-only rule generation, system auto-commits at max_turns.

    Returns AgentResult with all generated rules (deduplicated by diversity signature).
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
    corpus_id_label = f"corpus[{len(doc_contexts)}](diverse)"
    nav_turn_counter: int = 0

    for turn_index in range(max_turn_limit):
        if total_cost >= agent_config.budget_usd:
            return AgentResult(
                rules=discovered_rules, trajectory=[],
                total_cost_usd=total_cost, turns_used=turn_index,
                termination_reason="budget",
            )

        compact_hist = _compact_history(history, keep_recent=agent_config.history_compaction_threshold)
        stale_warning = _detect_stale_anchor(history, agent_config.stale_anchor_threshold)
        generated_summary = _format_generated_summary(discovered_rules)
        system_prompt = _build_diverse_system_prompt(query_text, corpus_toc, generated_summary)
        current_system = system_prompt + (stale_warning or "")
        if nav_turn_counter >= agent_config.nav_turn_cap:
            current_system += (
                f"\n\n[system] Navigation budget exhausted "
                f"({agent_config.nav_turn_cap} consecutive find_section calls). "
                "Next action MUST be `generate` (with a rule) or `batch_apply_rule`."
            )
        prompt = _serialize_prompt(current_system, compact_hist)

        try:
            _ensure_request_within_model_context(
                prompt_text=prompt,
                max_output_tokens=agent_config.agent_max_tokens,
                llm_provider=agent_config.agent_llm_provider,
                llm_model=agent_config.agent_llm_model,
                stage_label=f"diverse_agent q{query_idx} turn={turn_index}",
            )
        except RuntimeError as exc:
            print(f"    [turn {turn_index}] diverse prompt over context limit: {exc}")
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
            print(f"    [turn {turn_index}] diverse LLM call failed: {exc}")
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
                tool_result_full={"error": "invalid_json"},
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
                    tool_name="generate_invalid", tool_args={},
                    tool_result_preview=obs[:200],
                    cost_usd=agent_call_cost, latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"error": "invalid_rule_payload", "raw_rule": raw_rule},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            sig = _rule_diversity_signature(new_rule)
            if sig in seen_signatures:
                obs = (
                    f"REJECTED: rule duplicates an already-generated one on diversity "
                    f"signature {sig!r}. Vary mode / max_chars (small ≤200, medium ≤800, "
                    f"large >800) / anchor (first 50 chars). Already generated: {sorted(seen_signatures)!r}"
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
                    tool_args={"signature": list(sig)},
                    tool_result_preview=obs[:200],
                    cost_usd=agent_call_cost, latency_ms=agent_latency,
                    input_tokens=cache_result.input_tokens,
                    output_tokens=cache_result.output_tokens,
                    path_idx=path_idx,
                    tool_result_full={"duplicate_signature": list(sig),
                                      "already_generated_signatures": [list(s) for s in sorted(seen_signatures)]},
                    prompt_text=prompt,
                    raw_response=cache_result.response,
                )
                continue

            # Append-only commit
            discovered_rules.append(new_rule)
            seen_signatures.add(sig)
            obs = f"Rule R{len(discovered_rules)-1} accepted. Total accumulated: {len(discovered_rules)}."
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
                           "signature": list(sig)},
                tool_result_preview=obs[:200],
                cost_usd=agent_call_cost, latency_ms=agent_latency,
                input_tokens=cache_result.input_tokens,
                output_tokens=cache_result.output_tokens,
                path_idx=path_idx,
                tool_result_full={"rule_index": len(discovered_rules) - 1,
                                  "rule": new_rule.to_dict(),
                                  "total_accumulated": len(discovered_rules)},
                prompt_text=prompt,
                raw_response=cache_result.response,
            )
            continue

        # Tool dispatch (batch_apply_rule, find_section, apply_rule)
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

        history.append(_ConversationTurn(
            turn_index=turn_index,
            action_json=json.dumps(action, ensure_ascii=False),
            tool_name=tool_name, observation=obs_text,
        ))
        logger.log_turn(
            query_idx=query_idx, doc_id=corpus_id_label,
            turn_index=turn_index, agent_reasoning=reasoning,
            tool_name=tool_name, tool_args=tool_args,
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
        print(f"    [diverse t{turn_index}] {tool_name} → {obs_text[:80]}")

    # Reached max_turns: system auto-commits all accumulated rules
    return AgentResult(
        rules=discovered_rules, trajectory=[],
        total_cost_usd=total_cost, turns_used=max_turn_limit,
        termination_reason="max_turns",
    )


__all__ = [
    "_max_chars_bucket",
    "_rule_diversity_signature",
    "run_diverse_agent_on_query",
]
