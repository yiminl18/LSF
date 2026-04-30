"""Tool-Calling Agent CLI — two-phase experiment entry point.

Usage:
    PYTHONPATH=src python -m agent.tool_agent.cli \
        --config src/agent/config_pdfs_10doc.yaml \
        --queries 8,9 \
        --phase a \
        --max-docs 3 \
        --max-turns 10 \
        --agent-provider azure \
        --agent-model gpt-5.4-mini \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from agent.rule_runtime.data import get_label_filename
from agent.rule_runtime.holdout import (
    select_holdout_docs,
)
from agent.tool_agent.core import AgentConfig
from agent.tool_agent.hybrid import run_phase_a_hybrid  # kept: distinct control flow
from agent.tool_agent.orchestrator import run_phase_a, run_phase_b
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH

_DEFAULT_OUTPUT_ROOT = Path("output/agent/tool_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-Calling Agent experiment CLI")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--queries", default="8,9", help="Comma-separated query indices")
    parser.add_argument("--phase", choices=["a", "b", "both"], default="a", help="Which phase to run")
    parser.add_argument("--max-docs", type=int, default=10, help="Phase A max docs (0=unlimited)")
    parser.add_argument("--max-holdout-docs", type=int, default=25, help="Phase B max holdout docs")
    parser.add_argument("--holdout-seed", type=int, default=42, help="Seeded random Phase B holdout selection seed")
    parser.add_argument("--agent-provider", default="azure", help="Agent LLM provider")
    parser.add_argument("--agent-model", required=True, help="Agent LLM model")
    parser.add_argument("--eval-provider", default=None, help="Eval LLM provider (defaults to agent provider)")
    parser.add_argument("--eval-model", default=None, help="Eval LLM model (defaults to agent model)")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns per query")
    parser.add_argument("--budget", type=float, default=2.0, help="Budget per query per phase (USD)")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name", default="v1", help="Experiment name (used in output path)")
    parser.add_argument("--dry-run", action="store_true", help="Print config only, do not run")
    parser.add_argument(
        "--multipath-n",
        type=int,
        default=2,
        help="Number of paths in hybrid mode (default 2; ignored for single_shot or diverse)",
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=42,
        help="Seed for hybrid doc partitioning/shuffling (default 42; use multiple seeds for stability checks)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "single_shot",
            "diverse",
            "hybrid",
            "curriculum",
            "reflexion",
            "seq_cover",
        ],
        default="single_shot",
        help="Phase A mode: single_shot|diverse|hybrid|curriculum|reflexion|seq_cover",
    )
    args = parser.parse_args()
    if args.mode == "hybrid" and args.multipath_n < 2:
        parser.error(f"--mode hybrid requires --multipath-n >= 2, got: {args.multipath_n}")
    return args


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = config.get("dataset_root", "datasets/pdfs/latest")
    dataset_name = str(config.get("dataset", "pdfs"))
    # Parser field determines directory suffix (docling=default, mineru=_mineru isolated variant)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        processing_dir = Path(dataset_root) / "processing_mineru"
        label_dir = Path(dataset_root) / "label_mineru"
    elif parser == "docling":
        processing_dir = Path(dataset_root) / "processing"
        label_dir = Path(dataset_root) / "label"
    else:
        raise ValueError(f"Unsupported parser: {parser!r}, expected 'docling' or 'mineru'")
    truncate_before = config.get("truncate_before")
    eval_provider = args.eval_provider or args.agent_provider
    eval_model = args.eval_model or args.agent_model

    query_indices = [int(q) for q in args.queries.split(",")]

    agent_config = AgentConfig(
        max_turns_per_query=args.max_turns,
        budget_usd=args.budget,
        agent_llm_provider=args.agent_provider,
        agent_llm_model=args.agent_model,
    )

    print("=== Tool-Calling Agent ===")
    print(f"Config: {args.config}")
    print(f"Queries: {query_indices}")
    print(f"Phase: {args.phase}")
    print(f"Agent: {args.agent_provider}/{args.agent_model}")
    print(f"Eval: {eval_provider}/{eval_model}")
    print(f"Max docs (Phase A): {args.max_docs}")
    print(f"Max holdout docs (Phase B): {args.max_holdout_docs}")
    print(f"Holdout seed: {args.holdout_seed}")
    print(f"Max turns/query: {args.max_turns}")
    print(f"Budget/query/phase: ${args.budget}")
    print(f"Experiment: {args.experiment_name}")
    print()

    if args.dry_run:
        for qi in query_indices:
            query_config = None
            for qc in config.get("queries", []):
                if qc["query_idx"] == qi:
                    query_config = qc
                    break
            if query_config:
                docs = query_config.get("documents", [])
                if args.max_docs > 0:
                    docs = docs[:args.max_docs]
                print(f"q{qi} Phase A: {len(docs)} sampled docs")
                print(f"  {docs}")

                label_path = label_dir / get_label_filename({"dataset": dataset_name}, qi)
                sampled_set = set(query_config.get("documents", []))
                holdout = select_holdout_docs(
                    label_path, qi, processing_dir, sampled_set, args.max_holdout_docs,
                    seed=args.holdout_seed,
                )
                print(f"q{qi} Phase B: {len(holdout)} holdout docs")
                est_calls = len(docs) * args.max_turns + len(holdout) * 2
                print(f"  Est. max LLM calls: ~{est_calls}")
                print()
        return

    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)

    for qi in query_indices:
        print(f"\n{'='*60}")
        print(f"Query q{qi}")
        print(f"{'='*60}")

        query_config = None
        for qc in config.get("queries", []):
            if qc["query_idx"] == qi:
                query_config = qc
                break
        if query_config is None:
            print(f"  WARNING: q{qi} not found in config, skipping")
            continue

        doc_ids = query_config.get("documents", [])
        if args.max_docs > 0:
            doc_ids = doc_ids[:args.max_docs]

        # ---- Phase A -------------------------------------------------------
        if args.phase in ("a", "both"):
            print(f"\n--- Phase A: Sampled {len(doc_ids)} docs ---")
            t0 = time.time()
            phase_a_dir = args.output_root / f"q{qi}" / args.experiment_name / "phase_a"

            if args.mode == "hybrid":
                result_a = run_phase_a_hybrid(
                    query_idx=qi,
                    doc_ids=doc_ids,
                    processing_dir=processing_dir,
                    label_dir=label_dir,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    truncate_before=truncate_before,
                    cached_caller=cached_caller,
                    agent_config=agent_config,
                    output_dir=phase_a_dir,
                    n_paths=args.multipath_n,
                    partition_seed=args.partition_seed,
                )
            else:
                # single_shot (default) / diverse / curriculum
                result_a = run_phase_a(
                    query_idx=qi,
                    doc_ids=doc_ids,
                    processing_dir=processing_dir,
                    label_dir=label_dir,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    truncate_before=truncate_before,
                    cached_caller=cached_caller,
                    agent_config=agent_config,
                    output_dir=phase_a_dir,
                    mode=args.mode,
                )

            elapsed = time.time() - t0
            print(f"\n  Phase A done in {elapsed:.1f}s")
            print(f"  Rules: {len(result_a.best_rules)}, Cost: ${result_a.total_cost_usd:.4f}")
            if result_a.cross_doc_eval:
                best = result_a.cross_doc_eval[0]
                print(f"  Best rule: acc={best['accuracy']:.1%}, cov={best['coverage']:.1%}")

        # ---- Phase B -------------------------------------------------------
        if args.phase in ("b", "both"):
            print("\n--- Phase B: Holdout evaluation ---")
            phase_a_dir = args.output_root / f"q{qi}" / args.experiment_name / "phase_a"
            rules_path = phase_a_dir / "best_rules.json"

            if not rules_path.exists():
                print(f"  ERROR: Phase A rules not found at {rules_path}. Run phase A first.")
                continue

            phase_b_dir = args.output_root / f"q{qi}" / args.experiment_name / "phase_b"
            # Prefer reading the actual processed docs from the Phase A artifact to avoid
            # holdout contamination when --max-docs changes or Phase A silently skips a doc.
            # Falls back to CLI doc_ids only when the artifact is missing (old Phase A runs).
            phase_a_docs_path = phase_a_dir / "phase_a_docs.json"
            if phase_a_docs_path.exists():
                with phase_a_docs_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # Prefer excluded_doc_ids (all docs loaded by probe/eval, even if not directly
                # processed by agent); fall back to the older processed_doc_ids field.
                sampled_doc_ids = set(
                    data.get("excluded_doc_ids") or data.get("processed_doc_ids", [])
                )
                print(f"  Sampled exclusion list: {len(sampled_doc_ids)} docs (from {phase_a_docs_path.name})")
            else:
                print(f"  WARNING: {phase_a_docs_path.name} not found — using CLI doc_ids as fallback")
                sampled_doc_ids = set(doc_ids)
            t0 = time.time()
            report_b = run_phase_b(
                query_idx=qi,
                phase_a_rules_path=rules_path,
                sampled_doc_ids=sampled_doc_ids,
                processing_dir=processing_dir,
                label_dir=label_dir,
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                truncate_before=truncate_before,
                cached_caller=cached_caller,
                llm_provider=eval_provider,
                llm_model=eval_model,
                output_dir=phase_b_dir,
                max_holdout_docs=args.max_holdout_docs,
                holdout_seed=args.holdout_seed,
            )
            elapsed_b = time.time() - t0
            print(f"\n  Phase B done in {elapsed_b:.1f}s")
            print(f"  BestAcc={report_b['best_acc']:.1%}, UnionAcc={report_b['union_summary']['accuracy']:.1%}")
            gap = report_b.get('generalization_gap')
            if gap is not None:
                print(f"  Generalization gap: {gap:+.1%}")

    print("\n=== Done ===")
    print(f"Results: {args.output_root}/")


if __name__ == "__main__":
    main()
