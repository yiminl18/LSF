"""Agent orchestrator: Phase A (sampled exploration) + Phase B (holdout evaluation)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Literal

from agent.rule_runtime.data import get_label_filename, get_query_text
from agent.rule_runtime.deploy import evaluate_cascade
from agent.rule_runtime.holdout import (
    build_holdout_docs,
    build_holdout_report,
    run_per_rule_eval,
    run_union_eval,
    save_results,
    select_holdout_docs,
)
from agent.rule_runtime.artifacts import (
    load_rules_from_best_rules,
    load_sampled_eval_from_best_rules,
)
from agent.rules.range_rule_exec import execute_range_rule
from agent.rules.range_rule_json import RangeRule
from agent.rules.range_rule_scorer import score_retrieved_subset
from agent.tool_agent.core import (
    AgentConfig,
    AgentResult,
    run_agent_on_query,
)
from agent.tool_agent.code_core import (
    _cross_doc_evaluate_code,
    _dedupe_code_rules,
    _merge_rejection_counts,
    _rejection_counts_from_agent_result,
    _rejection_counts_from_eval,
    run_code_agent_on_query,
    save_code_phase_a_results,
)
from agent.tool_agent.curriculum_core import run_curriculum_agent_on_query
from agent.tool_agent.diverse_core import run_diverse_agent_on_query
from agent.tool_agent.document import DocumentContext, load_document_context
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.reflexion_core import run_reflexion_agent_on_query
from agent.tool_agent.rule_selection import select_best_rules
from agent.tool_agent.seq_cover_core import run_seq_cover_agent_on_query
from core.pipeline.e2e_utils.cache import CachedLLMCaller


# When success_count is tiny (e.g. 3), hint_reliability = 3/3 = 1.0 is over-fit
# and unsafe to trust for the holdout gate. Require at least this many judge-pass
# docs before computing reliability; otherwise keep it as None (gate disabled).
_MIN_SUPPORT_FOR_HINT_RELIABILITY = 5

PhaseAMode = Literal[
    "single_shot",
    "diverse",
    "curriculum",
    "reflexion",
    "seq_cover",
    "code",
]

_AGENT_FNS = {
    "single_shot": run_agent_on_query,
    "diverse": run_diverse_agent_on_query,
    "curriculum": run_curriculum_agent_on_query,
    "reflexion": run_reflexion_agent_on_query,
    "seq_cover": run_seq_cover_agent_on_query,
    "code": run_code_agent_on_query,
}

_AGENT_LABELS = {
    "single_shot": ("Per-Query Agent", "Agent"),
    "diverse": ("Diverse Per-Query Agent", "Diverse Agent"),
    "curriculum": ("Curriculum Per-Query Agent", "Curriculum Agent"),
    "reflexion": ("Reflexion Per-Query Agent", "Reflexion Agent"),
    "seq_cover": ("Seq-Cover Per-Query Agent", "Seq-Cover Agent"),
    "code": ("Code Per-Query Agent", "Code Agent"),
}


def _append_manifest(
    output_dir: Path,
    entry: dict[str, Any],
) -> None:
    """Append a cross-run entry to `<root>/manifest.jsonl`.

    output_dir is expected to look like `<root>/q{qi}/<exp>/phase_a/`; the
    manifest is written three levels up. If the depth does not match (e.g. unit
    tests with temp paths) the call is silently skipped.
    """
    import time as _time
    try:
        manifest_root = output_dir.parents[2]
    except IndexError:
        return
    manifest_path = manifest_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("timestamp", _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()))
    entry.setdefault("artifact_dir", str(output_dir.relative_to(manifest_root)))
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---- Phase A ----------------------------------------------------------------

@dataclass
class PhaseAResult:
    """Phase A run result."""

    query_idx: int
    best_rules: list[RangeRule]
    per_doc_agent_results: dict[str, AgentResult]  # doc_id -> AgentResult
    cross_doc_eval: list[dict[str, Any]]  # per-rule cross-document evaluation
    total_cost_usd: float


def _cross_doc_evaluate(
    rules: list[RangeRule],
    doc_contexts: list[DocumentContext],
    query_text: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    remaining_budget: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every rule on every document; return a ranked list.

    remaining_budget: if provided, stop processing further rules once the
    cumulative evaluation cost reaches the cap. Skipped rules still produce
    placeholder entries so rule_index is not fragmented.
    """
    rule_stats: list[dict[str, Any]] = []
    cum_eval_cost = 0.0
    budget_exhausted = False

    for ri, rule in enumerate(rules):
        if remaining_budget is not None and cum_eval_cost >= remaining_budget:
            if not budget_exhausted:
                print(
                    f"  [budget] cross_doc_eval budget ${remaining_budget:.4f} "
                    f"exhausted; skipping rule {ri}/{len(rules)} and beyond"
                )
                budget_exhausted = True
            rule_stats.append({
                "rule_index": ri,
                "rule_text": rule.rule_text[:80],
                "retrieval_spec": rule.retrieval_spec.to_dict(),
                "coverage": 0.0,
                "accuracy": 0.0,
                "score": 0.0,
                "matched_count": 0,
                "success_count": 0,
                "evaluated_docs": 0,
                "total_docs": len(doc_contexts),
                "budget_truncated": True,
                "eval_cost_usd": 0.0,
                "skipped_due_to_budget": True,
                "success_doc_ids": [],
                "matched_doc_ids": [],
                "answer_hint_pattern": rule.answer_hint_pattern,
                "phase_a_hint_reliability": None,
                "phase_a_hint_extraction_precision": None,
            })
            continue

        matched_count = 0
        success_count = 0
        total_cost = 0.0
        evaluated_doc_count = 0
        success_doc_ids: list[str] = []
        matched_doc_ids: list[str] = []
        hint_regex: re.Pattern[str] | None = None
        if rule.answer_hint_pattern:
            try:
                hint_regex = re.compile(rule.answer_hint_pattern)
            except re.error:
                hint_regex = None  # bad regex -> no hint signal, gate disabled
        hint_match_count = 0
        hint_match_and_judge_pass = 0
        # hint extraction precision guards against wide patterns like
        # \b[A-Z][a-z]+\b: P(regex match.group(0) in GT | hint_match AND judge_pass)
        hint_extraction_correct_count = 0

        for doc in doc_contexts:
            # Per-doc budget gate: a single rule's LLM calls can drain budget
            # fast, so check before each call rather than only around the rule.
            if remaining_budget is not None and cum_eval_cost >= remaining_budget:
                break
            evaluated_doc_count += 1
            subset = execute_range_rule(rule, doc.normalized_text)
            if not subset.matched:
                continue
            matched_count += 1
            matched_doc_ids.append(doc.doc_id)

            span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
            if not span_text.strip():
                continue

            score = score_retrieved_subset(
                question=query_text,
                retrieved_text=span_text,
                ground_truth=doc.ground_truth,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            if score.judge_result:
                success_count += 1
                success_doc_ids.append(doc.doc_id)
            if hint_regex is not None:
                m = hint_regex.search(span_text)
                if m is not None:
                    hint_match_count += 1
                    if score.judge_result:
                        hint_match_and_judge_pass += 1
                        gt_lower = (doc.ground_truth or "").lower()
                        if m.group(0).strip() and m.group(0).lower() in gt_lower:
                            hint_extraction_correct_count += 1
            gen_cost = score.metadata.get("generation", {}).get("cost_usd", 0.0)
            judge_cost = score.metadata.get("judge", {}).get("cost_usd", 0.0)
            total_cost += gen_cost + judge_cost
            cum_eval_cost += gen_cost + judge_cost

        doc_count = len(doc_contexts)
        # Score against the full sampled corpus so budget-truncated rules cannot
        # look perfect after only a few easy documents.
        denom = doc_count
        coverage = matched_count / denom if denom else 0.0
        accuracy = success_count / denom if denom else 0.0

        # Hint reliability is P(hint_match | judge=True) on sampled docs.
        # Only compute when success_count >= _MIN_SUPPORT_FOR_HINT_RELIABILITY
        # to avoid small-N over-fit (e.g. 3/3=1.0 triggering false negatives).
        hint_reliability: float | None = None
        hint_extraction_precision: float | None = None
        if hint_regex is not None and success_count >= _MIN_SUPPORT_FOR_HINT_RELIABILITY:
            hint_reliability = round(hint_match_and_judge_pass / success_count, 4)
            if hint_match_and_judge_pass > 0:
                hint_extraction_precision = round(
                    hint_extraction_correct_count / hint_match_and_judge_pass, 4
                )
            else:
                hint_extraction_precision = 0.0

        rule_stats.append({
            "rule_index": ri,
            "rule_text": rule.rule_text[:80],
            "retrieval_spec": rule.retrieval_spec.to_dict(),
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "score": round(coverage * accuracy, 4),
            "matched_count": matched_count,
            "success_count": success_count,
            "evaluated_docs": evaluated_doc_count,
            "total_docs": doc_count,
            "budget_truncated": evaluated_doc_count < doc_count,
            "eval_cost_usd": round(total_cost, 6),
            "success_doc_ids": success_doc_ids,
            "matched_doc_ids": matched_doc_ids,
            "answer_hint_pattern": rule.answer_hint_pattern,
            "hint_match_count": hint_match_count if hint_regex is not None else None,
            "hint_match_and_judge_pass": hint_match_and_judge_pass if hint_regex is not None else None,
            "phase_a_hint_reliability": hint_reliability,
            "phase_a_hint_extraction_precision": hint_extraction_precision,
        })

    rule_stats.sort(key=lambda x: x["score"], reverse=True)
    return rule_stats


_MAX_BEST_RULES = 5


def run_phase_a(
    query_idx: int,
    doc_ids: list[str],
    processing_dir: Path,
    label_dir: Path,
    dataset_root: str,
    truncate_before: str | None,
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    output_dir: Path,
    mode: PhaseAMode = "single_shot",
    dataset_name: str = "pdfs",
) -> PhaseAResult:
    """Phase A: explore rules on sampled docs using the selected agent mode."""
    if mode not in _AGENT_FNS:
        raise ValueError(f"unsupported Phase A mode: {mode!r}")
    agent_fn = _AGENT_FNS[mode]
    section_label, log_label = _AGENT_LABELS[mode]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrajectoryLogger(output_dir / "trajectory.jsonl")
    query_text = get_query_text(dataset_root, query_idx)

    print(f"  Loading {len(doc_ids)} documents...")
    doc_contexts: list[DocumentContext] = []
    for doc_id in doc_ids:
        try:
            ctx = load_document_context(
                doc_id, query_idx, processing_dir, label_dir,
                truncate_before=truncate_before,
                dataset_name=dataset_name,
            )
            doc_contexts.append(ctx)
        except (ValueError, FileNotFoundError) as e:
            print(f"  WARNING: skipping {doc_id}: {e}")
            continue

    # Any loaded doc may influence rule selection through the agent's cross-doc
    # probes or cross_doc_eval, so Phase B must exclude all of them even if
    # the budget caused some to drop out of the per-doc loop.
    excluded_doc_ids: list[str] = [c.doc_id for c in doc_contexts]

    print(
        f"\n  === {section_label} ({len(doc_contexts)} docs in corpus, "
        f"budget ${agent_config.budget_usd:.2f}) ==="
    )
    agent_result = agent_fn(
        query_text=query_text,
        query_idx=query_idx,
        doc_contexts=doc_contexts,
        cached_caller=cached_caller,
        agent_config=agent_config,
        logger=logger,
    )
    print(
        f"  {log_label}: {len(agent_result.rules)} rules, "
        f"cost=${agent_result.total_cost_usd:.4f}, "
        f"reason={agent_result.termination_reason}, "
        f"turns={agent_result.turns_used}"
    )

    if mode == "code":
        all_code_rules = list(agent_result.rules)
        processed_doc_ids = list(excluded_doc_ids)
        per_doc_results: dict[str, AgentResult] = {"__corpus__": agent_result}
        total_cost = agent_result.total_cost_usd
        unique_rules = _dedupe_code_rules(all_code_rules)

        processed_set = set(processed_doc_ids)
        selection_contexts = [c for c in doc_contexts if c.doc_id in processed_set]
        cross_doc_budget = max(0.0, agent_config.budget_usd - total_cost)
        print(
            f"\n  === Code Cross-Doc Evaluation ({len(unique_rules)} unique rules x "
            f"{len(selection_contexts)} selection docs, budget ${cross_doc_budget:.4f}) ==="
        )
        cross_doc_eval = _cross_doc_evaluate_code(
            unique_rules,
            selection_contexts,
            query_text,
            cached_caller,
            agent_config.agent_llm_provider,
            agent_config.agent_llm_model,
            remaining_budget=cross_doc_budget,
        )
        exploration_cost = total_cost
        cross_doc_eval_cost = sum(e.get("eval_cost_usd", 0.0) for e in cross_doc_eval)
        total_cost = exploration_cost + cross_doc_eval_cost
        print(
            f"  Costs: exploration ${exploration_cost:.4f} + "
            f"cross_doc_eval ${cross_doc_eval_cost:.4f} = total ${total_cost:.4f}"
        )

        best_rules = select_best_rules(
            cross_doc_eval,
            unique_rules,
            max_rules=_MAX_BEST_RULES,
            allow_no_score_first_three_fallback=False,
        )
        code_rule_rejections = _merge_rejection_counts(
            _rejection_counts_from_agent_result(agent_result),
            _rejection_counts_from_eval(cross_doc_eval),
        )
        save_code_phase_a_results(
            output_dir,
            query_idx,
            best_rules,
            cross_doc_eval,
            total_cost,
            processed_doc_ids,
            exploration_cost,
            cross_doc_eval_cost,
            excluded_doc_ids,
            code_rule_rejections,
        )
        summary_entry = {
            "query_idx": query_idx,
            "variant": mode,
            "num_paths": 1,
            "termination_reason": agent_result.termination_reason,
            "turns_used": agent_result.turns_used,
            "total_rules_discovered": len(all_code_rules),
            "unique_rules": len(unique_rules),
            "best_rules": len(best_rules),
            "code_rule_rejections": code_rule_rejections,
            "exploration_cost_usd": round(exploration_cost, 4),
            "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
            "total_cost_usd": round(total_cost, 4),
        }
        logger.log_summary(summary_entry)
        logger.close()

        manifest_entry = {k: summary_entry[k] for k in (
            "query_idx", "variant", "num_paths", "termination_reason", "turns_used",
            "total_rules_discovered", "unique_rules", "best_rules", "total_cost_usd",
        )}
        _append_manifest(output_dir, manifest_entry)

        return PhaseAResult(
            query_idx=query_idx,
            best_rules=best_rules,
            per_doc_agent_results=per_doc_results,
            cross_doc_eval=cross_doc_eval,
            total_cost_usd=total_cost,
        )

    all_rules: list[RangeRule] = list(agent_result.rules)
    processed_doc_ids: list[str] = list(excluded_doc_ids)
    per_doc_results: dict[str, AgentResult] = {"__corpus__": agent_result}
    total_cost = agent_result.total_cost_usd

    seen_specs: set[str] = set()
    unique_rules: list[RangeRule] = []
    for rule in all_rules:
        spec_key = json.dumps(rule.retrieval_spec.to_dict(), sort_keys=True)
        if spec_key not in seen_specs:
            seen_specs.add(spec_key)
            unique_rules.append(rule)

    # Only rank using docs the agent actually explored; including unprocessed
    # docs would leak into Phase B holdout selection.
    processed_set = set(processed_doc_ids)
    selection_contexts = [c for c in doc_contexts if c.doc_id in processed_set]

    cross_doc_budget = max(0.0, agent_config.budget_usd - total_cost)
    print(
        f"\n  === Cross-Doc Evaluation ({len(unique_rules)} unique rules x "
        f"{len(selection_contexts)} selection docs, budget ${cross_doc_budget:.4f}) ==="
    )
    cross_doc_eval = _cross_doc_evaluate(
        unique_rules, selection_contexts, query_text,
        cached_caller, agent_config.agent_llm_provider, agent_config.agent_llm_model,
        remaining_budget=cross_doc_budget,
    )

    # Cross-doc eval LLM cost must be part of Phase A total; otherwise the
    # reported Phase A cost covers only exploration (~1/2-1/5 of actual spend).
    exploration_cost = total_cost
    cross_doc_eval_cost = sum(e.get("eval_cost_usd", 0.0) for e in cross_doc_eval)
    total_cost = exploration_cost + cross_doc_eval_cost
    print(
        f"  Costs: exploration ${exploration_cost:.4f} + "
        f"cross_doc_eval ${cross_doc_eval_cost:.4f} = total ${total_cost:.4f}"
    )

    best_rules_raw = select_best_rules(cross_doc_eval, unique_rules, max_rules=_MAX_BEST_RULES)

    # Stamp computed hint reliability/extraction precision back onto best_rules.
    # RangeRule is frozen; use dataclasses.replace.
    spec_to_metrics: dict[str, tuple[float | None, float | None]] = {}
    for entry in cross_doc_eval:
        spec_key = json.dumps(entry["retrieval_spec"], sort_keys=True)
        spec_to_metrics[spec_key] = (
            entry.get("phase_a_hint_reliability"),
            entry.get("phase_a_hint_extraction_precision"),
        )

    best_rules: list[RangeRule] = []
    for r in best_rules_raw:
        spec_key = json.dumps(r.retrieval_spec.to_dict(), sort_keys=True)
        rel, prec = spec_to_metrics.get(spec_key, (None, None))
        if (rel is None and prec is None
                and r.phase_a_hint_reliability is None
                and r.phase_a_hint_extraction_precision is None):
            best_rules.append(r)
        else:
            best_rules.append(dc_replace(
                r,
                phase_a_hint_reliability=rel,
                phase_a_hint_extraction_precision=prec,
            ))

    _save_phase_a_results(
        output_dir, query_idx, best_rules, cross_doc_eval, total_cost,
        processed_doc_ids, exploration_cost, cross_doc_eval_cost,
        excluded_doc_ids,
    )
    summary_entry = {
        "query_idx": query_idx,
        "variant": mode,
        "num_paths": 1,
        "termination_reason": agent_result.termination_reason,
        "turns_used": agent_result.turns_used,
        "total_rules_discovered": len(all_rules),
        "unique_rules": len(unique_rules),
        "best_rules": len(best_rules),
        "exploration_cost_usd": round(exploration_cost, 4),
        "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    }
    logger.log_summary(summary_entry)
    logger.close()

    manifest_entry = {k: summary_entry[k] for k in (
        "query_idx", "variant", "num_paths", "termination_reason", "turns_used",
        "total_rules_discovered", "unique_rules", "best_rules", "total_cost_usd",
    )}
    _append_manifest(output_dir, manifest_entry)

    return PhaseAResult(
        query_idx=query_idx,
        best_rules=best_rules,
        per_doc_agent_results=per_doc_results,
        cross_doc_eval=cross_doc_eval,
        total_cost_usd=total_cost,
    )


def _save_phase_a_results(
    output_dir: Path,
    query_idx: int,
    best_rules: list[RangeRule],
    cross_doc_eval: list[dict[str, Any]],
    total_cost: float,
    processed_doc_ids: list[str],
    exploration_cost: float,
    cross_doc_eval_cost: float,
    excluded_doc_ids: list[str],
) -> None:
    """Write Phase A artifacts to disk."""
    rules_data = {
        "query_idx": query_idx,
        "packaging_mode": "tool_agent_phase_a",
        "merged_rules": [rule.to_dict() for rule in best_rules],
        "cross_doc_eval": cross_doc_eval,
        "exploration_cost_usd": round(exploration_cost, 4),
        "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    }
    with (output_dir / "best_rules.json").open("w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)

    # phase_a_docs.json: excluded_doc_ids is the authoritative exclusion set
    # for Phase B (every loaded doc, probe-touched or not). processed_doc_ids
    # is a subset recording docs actually handled by the per-doc loop.
    with (output_dir / "phase_a_docs.json").open("w", encoding="utf-8") as f:
        json.dump({
            "query_idx": query_idx,
            "excluded_doc_ids": excluded_doc_ids,
            "processed_doc_ids": processed_doc_ids,
            "count": len(excluded_doc_ids),
        }, f, indent=2, ensure_ascii=False)

    lines = [f"# Phase A Report: q{query_idx}\n"]
    lines.append(f"**Rules discovered**: {len(best_rules)}")
    lines.append(
        f"**Costs**: exploration ${exploration_cost:.4f} + "
        f"cross_doc_eval ${cross_doc_eval_cost:.4f} = total ${total_cost:.4f}\n"
    )
    lines.append("## Cross-Doc Evaluation\n")
    lines.append("| Rule | Coverage | Accuracy | Score | Matched | Success |")
    lines.append("|------|----------|----------|-------|---------|---------|")
    for e in cross_doc_eval:
        lines.append(
            f"| R{e['rule_index']} | {e['coverage']:.1%} | {e['accuracy']:.1%} "
            f"| {e['score']:.3f} | {e['matched_count']}/{e['total_docs']} "
            f"| {e['success_count']}/{e['total_docs']} |"
        )
    with (output_dir / "phase_a_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with (output_dir / "phase_a_report.json").open("w", encoding="utf-8") as f:
        json.dump({
            "query_idx": query_idx,
            "rule_count": len(best_rules),
            "cross_doc_eval": cross_doc_eval,
            "exploration_cost_usd": round(exploration_cost, 4),
            "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
            "total_cost_usd": round(total_cost, 4),
        }, f, indent=2, ensure_ascii=False)


# ---- Phase B ----------------------------------------------------------------


def _build_cascade_summary(
    cascade_rows: list[dict[str, Any]],
    n_rules: int,
) -> dict[str, Any]:
    """Aggregate cascade per-doc rows into a deployable metrics block."""
    n_docs = len(cascade_rows)
    if n_docs == 0:
        return {
            "policy": "cascade",
            "accuracy": 0.0,
            "success_docs": 0,
            "total_docs": 0,
            "total_cost_usd": 0.0,
            "avg_gen_calls_per_doc": 0.0,
            "avg_judge_calls_per_doc": 0.0,
            "rules_used_first_hit_distribution": {},
        }
    success = sum(1 for r in cascade_rows if r.get("judge_result") is True)
    total_cost = sum(r.get("actual_cost_usd", 0.0) for r in cascade_rows)
    total_gen = sum(r.get("gen_calls", 0) for r in cascade_rows)
    total_judge = sum(r.get("judge_calls", 0) for r in cascade_rows)
    # On judge-pass rows the final rules_used entry is the first successful rule.
    first_hit_dist: dict[int, int] = {}
    for r in cascade_rows:
        if r.get("judge_result") is True and r.get("rules_used"):
            first_hit = r["rules_used"][-1]
            first_hit_dist[first_hit] = first_hit_dist.get(first_hit, 0) + 1
    return {
        "policy": "cascade",
        "accuracy": success / n_docs,
        "success_docs": success,
        "total_docs": n_docs,
        "total_cost_usd": total_cost,
        "avg_gen_calls_per_doc": total_gen / n_docs,
        "avg_judge_calls_per_doc": total_judge / n_docs,
        "rules_used_first_hit_distribution": {str(k): v for k, v in sorted(first_hit_dist.items())},
        "n_rules_in_cascade": n_rules,
    }


def run_phase_b(
    query_idx: int,
    phase_a_rules_path: Path,
    sampled_doc_ids: set[str],
    processing_dir: Path,
    label_dir: Path,
    dataset_root: str,
    truncate_before: str | None,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    output_dir: Path,
    max_holdout_docs: int = 25,
    retrieval_too_large_token_threshold: int = 5000,
    holdout_seed: int = 42,
    dataset_name: str = "pdfs",
) -> dict[str, Any]:
    """Phase B: evaluate Phase A rules on holdout docs (per-rule + union + cascade)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = load_rules_from_best_rules(phase_a_rules_path)
    if not rules:
        raise ValueError(f"Phase A rules file is empty or invalid: {phase_a_rules_path}")

    query_text = get_query_text(dataset_root, query_idx)

    label_path = label_dir / get_label_filename({"dataset": dataset_name}, query_idx)
    holdout_ids = select_holdout_docs(
        label_path, query_idx, processing_dir, sampled_doc_ids, max_holdout_docs,
        seed=holdout_seed,
    )
    print(f"  Holdout docs: {len(holdout_ids)}")

    docs = build_holdout_docs(holdout_ids, query_idx, processing_dir, label_path, truncate_before)
    print(f"  Loaded {len(docs)} docs, total tokens: {sum(d.token_count for d in docs)}")

    print("  === Per-Rule Evaluation ===")
    per_rule_rows = run_per_rule_eval(
        query_idx, query_text, rules, docs,
        cached_caller, llm_provider, llm_model,
        retrieval_too_large_token_threshold,
        packaging_mode="tool_agent_phase_b",
    )

    print("  === Union Evaluation ===")
    union_rows = run_union_eval(
        query_idx, query_text, rules, docs,
        cached_caller, llm_provider, llm_model,
        retrieval_too_large_token_threshold,
        packaging_mode="tool_agent_phase_b",
    )

    # Cascade is the deployable metric; per_rule is oracle and union is a
    # weak-deployable upper bound.
    print("  === Cascade Evaluation ===")
    sampled_eval = load_sampled_eval_from_best_rules(phase_a_rules_path, rules)
    cascade_rows = evaluate_cascade(
        rules, sampled_eval, docs, query_idx, query_text,
        cached_caller, llm_provider, llm_model,
        retrieval_too_large_token_threshold,
    )
    cascade_summary = _build_cascade_summary(cascade_rows, n_rules=len(rules))
    print(
        f"  Cascade: acc={cascade_summary['accuracy']:.3f}, "
        f"cost=${cascade_summary['total_cost_usd']:.4f}, "
        f"avg_gen={cascade_summary['avg_gen_calls_per_doc']:.2f}/doc, "
        f"avg_judge={cascade_summary['avg_judge_calls_per_doc']:.2f}/doc"
    )

    report = build_holdout_report(
        query_idx, query_text, per_rule_rows, union_rows, rules, None,
    )
    report["cascade_summary"] = cascade_summary

    save_results(output_dir, per_rule_rows, union_rows, report)
    cascade_path = output_dir / "holdout_cascade_rows.jsonl"
    with cascade_path.open("w", encoding="utf-8") as f:
        for row in cascade_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report
