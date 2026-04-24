"""Focused runner for agent experiments.

This module is a thin orchestration layer over the existing experiment entry
points. It intentionally runs one selected experiment at a time; aggregate
presets such as "all" are not supported.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Iterator, Sequence

import yaml

from agent.reflection_agent.runner import run_baseline_sweep
from agent.rule_runtime import deploy as deploy_module
from agent.rule_runtime import holdout as holdout_module
from agent.rule_runtime.holdout import select_holdout_docs
from agent.tool_agent import cli as tool_agent_cli


_DEFAULT_CONFIG = Path("src/agent/config_pdfs_10doc.yaml")
_DEFAULT_BUNDLE_OUTPUT_ROOT = Path("output/agent/financial_baseline_runner")
_DEFAULT_TOOL_OUTPUT_ROOT = Path("output/agent/tool_agent")

_BUNDLE_EXPERIMENTS = {
    "bundle-full": "full_bundle_reference",
    "bundle-grouped": "grouped_433",
}

_TOOL_AGENT_EXPERIMENTS = {
    "tool-agent-trivial": "single_shot",
    "tool-agent-diverse": "diverse",
    "tool-agent-hybrid": "hybrid",
    "tool-agent-curriculum": "curriculum",
}

_EXPERIMENT_CHOICES = tuple(_BUNDLE_EXPERIMENTS) + tuple(_TOOL_AGENT_EXPERIMENTS)


@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


def _parse_query_indices(raw: str) -> list[int]:
    try:
        query_indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --queries value: {raw!r}") from exc
    if not query_indices:
        raise argparse.ArgumentTypeError("--queries must include at least one query index")
    return query_indices


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return loaded


def _query_docs(config: dict, query_idx: int) -> list[str]:
    for query_config in config.get("queries", []):
        if int(query_config.get("query_idx")) == query_idx:
            return list(query_config.get("documents", []) or [])
    return []


def _processing_and_label_dirs(config: dict) -> tuple[Path, Path]:
    dataset_root = config.get("dataset_root", "datasets/pdfs/latest")
    parser_name = config.get("parser", "docling")
    if parser_name == "docling":
        return Path(dataset_root) / "processing", Path(dataset_root) / "label"
    if parser_name == "mineru":
        return Path(dataset_root) / "processing_mineru", Path(dataset_root) / "label_mineru"
    raise ValueError(f"Unsupported parser: {parser_name!r}; expected 'docling' or 'mineru'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one focused agent experiment: bundle baseline or tool-agent variant."
    )
    parser.add_argument("--experiment", required=True, choices=_EXPERIMENT_CHOICES)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--queries", default="3", help="Comma-separated query indices")
    parser.add_argument("--phase", choices=["a", "b", "both"], default="both")
    parser.add_argument("--agent-provider", default="azure")
    parser.add_argument("--agent-model", default="gpt-5.4-mini")
    parser.add_argument("--eval-provider", default=None)
    parser.add_argument("--eval-model", default=None)
    parser.add_argument("--max-docs", type=int, default=10, help="Tool-agent Phase A max docs")
    parser.add_argument("--max-holdout-docs", type=int, default=25)
    parser.add_argument("--holdout-seed", type=int, default=42)
    parser.add_argument("--bundle-output-root", type=Path, default=_DEFAULT_BUNDLE_OUTPUT_ROOT)
    parser.add_argument("--tool-output-root", type=Path, default=_DEFAULT_TOOL_OUTPUT_ROOT)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--multipath-n", type=int, default=2)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument(
        "--bundle-deploy",
        action="store_true",
        help="Run bundle cascade/deploy after bundle Phase B.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without LLM calls")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _parse_query_indices(args.queries)

    is_bundle = args.experiment in _BUNDLE_EXPERIMENTS
    if args.bundle_deploy and not is_bundle:
        parser.error("--bundle-deploy is only valid for bundle experiments")
    if args.bundle_deploy and args.phase == "a":
        parser.error("--bundle-deploy requires --phase b or --phase both")
    if args.experiment == "tool-agent-hybrid" and args.multipath_n < 2:
        parser.error("--experiment tool-agent-hybrid requires --multipath-n >= 2")
    return args


def _run_bundle_phase_a(args: argparse.Namespace, packaging_mode: str) -> None:
    query_indices = _parse_query_indices(args.queries)
    if args.dry_run:
        config = _load_config(args.config)
        print("=== Bundle Phase A dry-run ===")
        print(f"Packaging mode: {packaging_mode}")
        print(f"Config: {args.config}")
        print(f"Queries: {query_indices}")
        print(f"Output root: {args.bundle_output_root}")
        for query_idx in query_indices:
            docs = _query_docs(config, query_idx)
            grouped_note = ""
            if packaging_mode == "grouped_433" and len(docs) != 10:
                grouped_note = " (grouped_433 expects exactly 10 docs)"
            print(f"  q{query_idx}: {len(docs)} sampled docs{grouped_note}")
        return

    run_baseline_sweep(
        packaging_modes=(packaging_mode,),
        query_indices=query_indices,
        config_path=args.config,
        output_root=args.bundle_output_root,
        llm_provider=args.agent_provider,
        llm_model=args.agent_model,
    )


def _run_bundle_phase_b(args: argparse.Namespace, packaging_mode: str) -> None:
    eval_provider = args.eval_provider or args.agent_provider
    eval_model = args.eval_model or args.agent_model
    if args.dry_run:
        config = _load_config(args.config)
        processing_dir, label_dir = _processing_and_label_dirs(config)
        print("=== Bundle Phase B dry-run ===")
        print(f"Packaging mode: {packaging_mode}")
        print(f"Config: {args.config}")
        print(f"Queries: {_parse_query_indices(args.queries)}")
        print(f"Output root: {args.bundle_output_root}")
        print(f"Eval: {eval_provider}/{eval_model}")
        for query_idx in _parse_query_indices(args.queries):
            sampled_docs = set(_query_docs(config, query_idx))
            label_path = label_dir / f"10k_q{query_idx}_reconstructed_labels.json"
            holdout = select_holdout_docs(
                label_path=label_path,
                query_idx=query_idx,
                processing_dir=processing_dir,
                sampled_doc_ids=sampled_docs,
                max_docs=args.max_holdout_docs,
                seed=args.holdout_seed,
            )
            best_rules = (
                args.bundle_output_root / f"q{query_idx}" / packaging_mode / "best_rules.json"
            )
            print(
                f"  q{query_idx}: {len(sampled_docs)} sampled docs, "
                f"{len(holdout)} holdout docs, rules={best_rules}"
            )
        return

    argv = [
        "agent.rule_runtime.holdout",
        "--config",
        str(args.config),
        "--packaging-mode",
        packaging_mode,
        "--queries",
        args.queries,
        "--max-docs",
        str(args.max_holdout_docs),
        "--holdout-seed",
        str(args.holdout_seed),
        "--llm-provider",
        eval_provider,
        "--llm-model",
        eval_model,
        "--output-root",
        str(args.bundle_output_root),
    ]
    with _temporary_argv(argv):
        holdout_module.main()


def _run_bundle_deploy(args: argparse.Namespace, packaging_mode: str) -> None:
    eval_provider = args.eval_provider or args.agent_provider
    eval_model = args.eval_model or args.agent_model
    query_indices = _parse_query_indices(args.queries)

    if args.dry_run:
        print("=== Bundle deploy dry-run ===")
        print(f"Packaging mode: {packaging_mode}")
        for query_idx in query_indices:
            best_rules = (
                args.bundle_output_root / f"q{query_idx}" / packaging_mode / "best_rules.json"
            )
            output_dir = args.bundle_output_root / f"q{query_idx}" / packaging_mode / "deploy"
            print(
                f"  q{query_idx}: deploy {best_rules} -> {output_dir} "
                f"({args.max_holdout_docs} holdout docs)"
            )
        return

    for query_idx in query_indices:
        best_rules = args.bundle_output_root / f"q{query_idx}" / packaging_mode / "best_rules.json"
        output_dir = args.bundle_output_root / f"q{query_idx}" / packaging_mode / "deploy"
        argv = [
            "agent.rule_runtime.deploy",
            "--query-idx",
            str(query_idx),
            "--config",
            str(args.config),
            "--in-best-rules",
            str(best_rules),
            "--output-dir",
            str(output_dir),
            "--max-holdout-docs",
            str(args.max_holdout_docs),
            "--holdout-seed",
            str(args.holdout_seed),
            "--llm-provider",
            eval_provider,
            "--llm-model",
            eval_model,
        ]
        with _temporary_argv(argv):
            deploy_module.main()


def _run_tool_agent(args: argparse.Namespace, mode: str) -> None:
    argv = [
        "agent.tool_agent.cli",
        "--config",
        str(args.config),
        "--queries",
        args.queries,
        "--phase",
        args.phase,
        "--max-docs",
        str(args.max_docs),
        "--max-holdout-docs",
        str(args.max_holdout_docs),
        "--holdout-seed",
        str(args.holdout_seed),
        "--agent-provider",
        args.agent_provider,
        "--agent-model",
        args.agent_model,
        "--max-turns",
        str(args.max_turns),
        "--budget",
        str(args.budget),
        "--output-root",
        str(args.tool_output_root),
        "--experiment-name",
        args.experiment,
        "--mode",
        mode,
        "--multipath-n",
        str(args.multipath_n),
        "--partition-seed",
        str(args.partition_seed),
    ]
    if args.eval_provider is not None:
        argv.extend(["--eval-provider", args.eval_provider])
    if args.eval_model is not None:
        argv.extend(["--eval-model", args.eval_model])
    if args.dry_run:
        argv.append("--dry-run")

    with _temporary_argv(argv):
        tool_agent_cli.main()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.experiment in _BUNDLE_EXPERIMENTS:
        packaging_mode = _BUNDLE_EXPERIMENTS[args.experiment]
        if args.phase in ("a", "both"):
            _run_bundle_phase_a(args, packaging_mode)
        if args.phase in ("b", "both"):
            _run_bundle_phase_b(args, packaging_mode)
            if args.bundle_deploy:
                _run_bundle_deploy(args, packaging_mode)
        return

    mode = _TOOL_AGENT_EXPERIMENTS[args.experiment]
    _run_tool_agent(args, mode)


if __name__ == "__main__":
    main()
