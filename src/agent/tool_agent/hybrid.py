"""Hybrid Phase A — multipath × diverse: per-path doc subset + per-path doc order + diverse agent.

Algorithm:
  1. _partition_overlapping splits doc_contexts into N subsets (each with path-specific
     set membership and doc ordering).
  2. Each subset runs run_diverse_agent_on_query independently.
  3. Union all path rules, deduplicating by exact retrieval_spec JSON key.
  4. Run _cross_doc_evaluate on the full doc_contexts over the deduplicated rule pool.
  5. select_best_rules (set-cover) picks best_rules; artifacts match the run_phase_a format.

Cost: N paths × diverse-mode Phase A budget (roughly $0.4/path → N=2 roughly $0.8).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

from agent.rule_runtime.data import get_query_text
from agent.rules.range_rule_json import RangeRule
from agent.tool_agent.core import AgentConfig, AgentResult
from agent.tool_agent.diverse_core import run_diverse_agent_on_query
from agent.tool_agent.document import DocumentContext, load_document_context
from agent.tool_agent.logger import TrajectoryLogger
from agent.tool_agent.orchestrator import (
    PhaseAResult,
    _append_manifest,
    _cross_doc_evaluate,
    _save_phase_a_results,
)
from agent.tool_agent.rule_selection import select_best_rules
from core.pipeline.e2e_utils.cache import CachedLLMCaller


def _partition_overlapping(
    doc_ids: list[str],
    n_paths: int,
    seed: int = 42,
) -> list[list[str]]:
    """Split doc_ids into n_paths overlapping subsets; each doc appears in roughly ceil(n_paths * 2/3) subsets.

    Each subset's membership is determined by a shared rng_subset; each subset's ordering
    is shuffled by a path-specific rng_order (seed + path_idx + 1) so each path agent
    encounters different first-turn anchors.

    n_paths=1: no partitioning. n_paths=2: each subset ~75% of docs. n_paths=3: Latin-square.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths}")

    if n_paths == 1:
        return [list(doc_ids)]

    n_docs = len(doc_ids)
    if n_docs == 0:
        return [[] for _ in range(n_paths)]

    rng_subset = random.Random(seed)

    if n_paths == 2:
        target_per_subset = max(1, round(n_docs * 0.75))
        subsets: list[list[str]] = []
        for path_idx in range(n_paths):
            shuffled = list(doc_ids)
            rng_subset.shuffle(shuffled)
            subset = shuffled[:target_per_subset]
            rng_order = random.Random(seed + path_idx + 1)
            rng_order.shuffle(subset)
            subsets.append(subset)
        return subsets

    if n_paths == 3:
        shuffled = list(doc_ids)
        rng_subset.shuffle(shuffled)
        chunk = (n_docs + 2) // 3
        groups: list[list[str]] = [
            shuffled[i * chunk : (i + 1) * chunk] for i in range(3)
        ]
        subsets = []
        for path_idx in range(3):
            excluded = set(groups[path_idx])
            subset = [d for d in shuffled if d not in excluded]
            rng_order = random.Random(seed + path_idx + 1)
            rng_order.shuffle(subset)
            subsets.append(subset)
        return subsets

    # n_paths > 3: each doc appears in ceil(n_paths * 2/3) paths
    appearances = math.ceil(n_paths * 2 / 3)
    pool = doc_ids * appearances
    rng_subset.shuffle(pool)
    per_path = len(pool) // n_paths
    result: list[list[str]] = []
    for path_idx in range(n_paths):
        chunk_docs = pool[path_idx * per_path : (path_idx + 1) * per_path]
        seen: set[str] = set()
        deduped: list[str] = []
        for d in chunk_docs:
            if d not in seen:
                seen.add(d)
                deduped.append(d)
        rng_order = random.Random(seed + path_idx + 1)
        rng_order.shuffle(deduped)
        result.append(deduped)
    return result


def run_phase_a_hybrid(
    query_idx: int,
    doc_ids: list[str],
    processing_dir: Path,
    label_dir: Path,
    dataset_root: str,
    truncate_before: str | None,
    cached_caller: CachedLLMCaller,
    agent_config: AgentConfig,
    output_dir: Path,
    n_paths: int = 2,
    partition_seed: int = 42,
    dataset_name: str = "pdfs",
) -> PhaseAResult:
    """Phase A (hybrid): N-path diverse agent exploration + union dedup + set-cover.

    Each path receives a distinct doc subset and ordering, widening the candidate pool
    across both breadth (multiple subsets) and depth (per-path diversity generation).

    Args:
        query_idx: Query index.
        doc_ids: Phase A sampled doc IDs.
        processing_dir: Directory of processed JSON files.
        label_dir: Label directory.
        dataset_root: Dataset root path.
        truncate_before: Truncation marker (or None).
        cached_caller: Cached LLM caller.
        agent_config: Agent runtime config.
        output_dir: Output directory for artifacts.
        n_paths: Number of independent paths (default 2).
        partition_seed: Partition RNG seed (default 42).

    Returns:
        PhaseAResult in the same format as run_phase_a.
    """
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

    excluded_doc_ids: list[str] = [c.doc_id for c in doc_contexts]

    loaded_ids = [c.doc_id for c in doc_contexts]
    subsets_ids = _partition_overlapping(loaded_ids, n_paths, seed=partition_seed)
    id_to_ctx: dict[str, DocumentContext] = {c.doc_id: c for c in doc_contexts}

    print(
        f"\n  === Hybrid Phase A: {n_paths} paths × diverse agent, "
        f"{len(doc_contexts)} total docs, seed={partition_seed} ==="
    )

    all_rules_per_path: list[list[RangeRule]] = []
    per_path_costs: list[float] = []
    per_path_termination: list[str] = []
    total_exploration_cost = 0.0

    for path_i, subset_ids in enumerate(subsets_ids):
        # Preserve path-specific ordering set by _partition_overlapping
        subset_ctxs = [id_to_ctx[d] for d in subset_ids if d in id_to_ctx]
        print(
            f"\n  [path {path_i}] {len(subset_ctxs)} docs (diverse), "
            f"budget=${agent_config.budget_usd:.2f}"
        )
        agent_result = run_diverse_agent_on_query(
            query_text=query_text,
            query_idx=query_idx,
            doc_contexts=subset_ctxs,
            cached_caller=cached_caller,
            agent_config=agent_config,
            logger=logger,
            path_idx=path_i,
        )
        print(
            f"  [path {path_i}] {len(agent_result.rules)} rules, "
            f"cost=${agent_result.total_cost_usd:.4f}, "
            f"reason={agent_result.termination_reason}"
        )
        all_rules_per_path.append(agent_result.rules)
        per_path_costs.append(agent_result.total_cost_usd)
        per_path_termination.append(agent_result.termination_reason)
        total_exploration_cost += agent_result.total_cost_usd

    per_path_rule_counts = [len(r) for r in all_rules_per_path]

    # Union + exact dedup by retrieval_spec JSON key
    seen_specs: set[str] = set()
    union_rules: list[RangeRule] = []
    for path_rules in all_rules_per_path:
        for rule in path_rules:
            spec_key = json.dumps(rule.retrieval_spec.to_dict(), sort_keys=True)
            if spec_key not in seen_specs:
                seen_specs.add(spec_key)
                union_rules.append(rule)

    deduped_count = len(union_rules)
    print(
        f"\n  Union: {sum(per_path_rule_counts)} total rules → "
        f"{deduped_count} unique after dedup"
    )

    # Cross-doc eval uses the full doc_contexts, not per-path subsets
    cross_doc_budget = max(0.0, agent_config.budget_usd - total_exploration_cost)
    print(
        f"\n  === Cross-Doc Evaluation ({deduped_count} unique rules × "
        f"{len(doc_contexts)} docs, budget ${cross_doc_budget:.4f}) ==="
    )
    cross_doc_eval = _cross_doc_evaluate(
        union_rules, doc_contexts, query_text,
        cached_caller, agent_config.agent_llm_provider, agent_config.agent_llm_model,
        remaining_budget=cross_doc_budget,
    )

    cross_doc_eval_cost = sum(e.get("eval_cost_usd", 0.0) for e in cross_doc_eval)
    total_cost = total_exploration_cost + cross_doc_eval_cost
    print(
        f"  Costs: exploration ${total_exploration_cost:.4f} + "
        f"cross_doc_eval ${cross_doc_eval_cost:.4f} = total ${total_cost:.4f}"
    )

    MAX_BEST_RULES = 5
    best_rules_raw = select_best_rules(cross_doc_eval, union_rules, max_rules=MAX_BEST_RULES)

    # Stamp hint reliability/precision back onto best_rules from cross_doc_eval metrics
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
        if (
            rel is None and prec is None
            and r.phase_a_hint_reliability is None
            and r.phase_a_hint_extraction_precision is None
        ):
            best_rules.append(r)
        else:
            best_rules.append(_dc_replace(
                r,
                phase_a_hint_reliability=rel,
                phase_a_hint_extraction_precision=prec,
            ))

    hybrid_meta: dict[str, Any] = {
        "n_paths": n_paths,
        "partition_seed": partition_seed,
        "subsets": subsets_ids,
        "per_path_rules_count": per_path_rule_counts,
        "deduped_unique_rules": deduped_count,
    }

    # Persist using _save_phase_a_results, then append hybrid_meta
    _save_phase_a_results(
        output_dir, query_idx, best_rules, cross_doc_eval, total_cost,
        excluded_doc_ids, total_exploration_cost, cross_doc_eval_cost,
        excluded_doc_ids,
    )

    rules_path = output_dir / "best_rules.json"
    with rules_path.open("r", encoding="utf-8") as f:
        rules_data = json.load(f)
    rules_data["hybrid_meta"] = hybrid_meta
    with rules_path.open("w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)

    logger.log_summary({
        "query_idx": query_idx,
        "variant": "hybrid",
        "n_paths": n_paths,
        "num_paths": n_paths,
        "per_path_rules_count": per_path_rule_counts,
        "per_path_termination": per_path_termination,
        "deduped_unique_rules": deduped_count,
        "best_rules": len(best_rules),
        "exploration_cost_usd": round(total_exploration_cost, 4),
        "cross_doc_eval_cost_usd": round(cross_doc_eval_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    })
    logger.close()

    _append_manifest(output_dir, {
        "query_idx": query_idx,
        "variant": "hybrid",
        "num_paths": n_paths,
        "per_path_termination": per_path_termination,
        "per_path_rules_count": per_path_rule_counts,
        "deduped_unique_rules": deduped_count,
        "best_rules": len(best_rules),
        "total_cost_usd": round(total_cost, 4),
    })

    per_doc_results: dict[str, AgentResult] = {}

    return PhaseAResult(
        query_idx=query_idx,
        best_rules=best_rules,
        per_doc_agent_results=per_doc_results,
        cross_doc_eval=cross_doc_eval,
        total_cost_usd=total_cost,
    )
