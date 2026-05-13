"""Sweep runner for paper baselines.

run_baseline_sweep(experiment, ...) iterates over (query_idx, doc_id)
pairs (same doc set as config), calls the appropriate extractor, scores
the result, and writes baseline_rows.jsonl + baseline_summary.json.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs
from agent.baselines.loader import build_doc_inputs, load_config
from agent.baselines.scorer_adapter import error_row, score_and_build_row
from agent.rule_runtime.data import get_query_text
from agent.rule_runtime.deploy import DeployedRow, summarize_rows
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH


_DEFAULT_BASELINE_OUTPUT_ROOT = Path("output/agent/baselines")

logger = logging.getLogger(__name__)


def _query_doc_ids(config: dict[str, Any], query_idx: int) -> list[str]:
    for qc in config.get("queries", []):
        if int(qc.get("query_idx", -1)) == query_idx:
            return list(qc.get("documents", []) or [])
    return []


def _get_extractor(
    experiment: str,
    deepread_max_pages: int | None = None,
    deepread_ocr_model: str | None = None,
    deepread_ocr_provider: str | None = None,
) -> BaselineExtractor:
    if experiment == "exit":
        from agent.baselines.exit.extractor import ExitExtractor
        return ExitExtractor()
    if experiment == "deepread":
        from agent.baselines.deepread.extractor import DeepReadExtractor
        kwargs: dict[str, Any] = {}
        if deepread_max_pages is not None:
            kwargs["max_pages"] = deepread_max_pages
        if deepread_ocr_model is not None:
            kwargs["ocr_model"] = deepread_ocr_model
        if deepread_ocr_provider is not None:
            kwargs["ocr_provider"] = deepread_ocr_provider
        return DeepReadExtractor(**kwargs)
    if experiment == "mdocagent":
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor
        return MDocAgentExtractor()
    raise ValueError(f"Unknown baseline experiment: {experiment!r}")


def run_baseline_sweep(
    experiment: str,
    query_indices: list[int],
    config_path: Path,
    output_root: Path = _DEFAULT_BASELINE_OUTPUT_ROOT,
    llm_provider: str = "azure",
    llm_model: str = "gpt-5.4-mini",
    eval_provider: str | None = None,
    eval_model: str | None = None,
    dry_run: bool = False,
    max_docs: int | None = None,
    deepread_max_pages: int | None = None,
    deepread_ocr_model: str | None = None,
    deepread_ocr_provider: str | None = None,
) -> None:
    """Run the baseline sweep and write results to output_root/<experiment>/q<idx>/."""
    config = load_config(config_path)
    eval_provider = eval_provider or llm_provider
    eval_model = eval_model or llm_model

    extractor = _get_extractor(
        experiment,
        deepread_max_pages=deepread_max_pages,
        deepread_ocr_model=deepread_ocr_model,
        deepread_ocr_provider=deepread_ocr_provider,
    )
    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)
    dataset_root = config.get("dataset_root", "datasets/pdfs/latest")

    for query_idx in query_indices:
        query_text = get_query_text(dataset_root, query_idx)
        doc_ids = _query_doc_ids(config, query_idx)
        if not doc_ids:
            print(f"[baseline-{experiment}] q{query_idx}: no docs configured, skipping")
            continue

        if max_docs is not None:
            doc_ids = doc_ids[:max_docs]
            logger.info("Capped doc set to %d docs (--max-docs)", max_docs)

        out_dir = output_root / experiment / f"q{query_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[baseline-{experiment}] q{query_idx}: {len(doc_ids)} docs, "
            f"provider={llm_provider}/{llm_model}"
        )

        if dry_run:
            print(f"  [dry-run] would process: {doc_ids}")
            continue

        rows: list[DeployedRow] = []
        for doc_id in doc_ids:
            t0 = time.perf_counter()
            try:
                doc_inputs: DocInputs = build_doc_inputs(config, query_idx, doc_id)
                result = extractor.extract(
                    query_idx=query_idx,
                    query_text=query_text,
                    doc_id=doc_id,
                    doc_inputs=doc_inputs,
                    cached_caller=cached_caller,
                )
                row = score_and_build_row(
                    query_idx=query_idx,
                    doc_id=doc_id,
                    policy=f"baseline-{experiment}",
                    result=result,
                    ground_truth=doc_inputs.ground_truth,
                    query_text=query_text,
                    cached_caller=cached_caller,
                    llm_provider=eval_provider,
                    llm_model=eval_model,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"  [baseline-{experiment}] q{query_idx}/{doc_id} FAILED: {exc}")
                print(tb)
                row = error_row(
                    query_idx=query_idx,
                    doc_id=doc_id,
                    policy=f"baseline-{experiment}",
                    blocker=f"extractor_error:{type(exc).__name__}:{exc}",
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            status = "pass" if row["judge_result"] is True else "fail"
            print(
                f"  {doc_id}: {status} cost=${row['actual_cost_usd']:.4f} "
                f"latency={elapsed_ms:.0f}ms"
            )
            rows.append(row)

        # Write rows and summary
        rows_path = out_dir / "baseline_rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

        summary = summarize_rows(rows, policy_name=f"baseline-{experiment}")
        summary_path = out_dir / "baseline_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            f"[baseline-{experiment}] q{query_idx} done: "
            f"acc={summary['deployed_acc']:.1%} cost=${summary['total_cost_usd']:.4f}"
        )
        print(f"  rows -> {rows_path}")
        print(f"  summary -> {summary_path}")
