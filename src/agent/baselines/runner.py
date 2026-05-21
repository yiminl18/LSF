"""Sweep runner for paper baselines.

run_baseline_sweep(experiment, ...) iterates over (query_idx, doc_id)
pairs (same doc set as config), calls the appropriate extractor, scores
the result, and writes baseline_rows.jsonl + baseline_summary.json.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs
from agent.baselines.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
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


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._-") or "dataset"


def _dataset_output_key(config: dict[str, Any]) -> str:
    dataset = config.get("dataset")
    if dataset:
        return _safe_path_component(str(dataset))
    from agent.baselines.defaults import resolve_dataset_root

    dataset_root = resolve_dataset_root(config)
    if dataset_root.name == "latest" and dataset_root.parent.name:
        return _safe_path_component(dataset_root.parent.name)
    return _safe_path_component(dataset_root.name)


def _get_extractor(
    experiment: str,
    deepread_max_pages: int | None = None,
    deepread_ocr_model: str | None = None,
    deepread_ocr_provider: str | None = None,
    mdocagent_max_pages: int | None = None,
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
        return MDocAgentExtractor(max_pages=mdocagent_max_pages)
    if experiment == "qa-agent":
        from agent.baselines.qa_agent.extractor import QAAgentExtractor
        return QAAgentExtractor()
    raise ValueError(f"Unknown baseline experiment: {experiment!r}")


def run_baseline_sweep(
    experiment: str,
    query_indices: list[int],
    config_path: Path,
    output_root: Path = _DEFAULT_BASELINE_OUTPUT_ROOT,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    eval_provider: str | None = None,
    eval_model: str | None = None,
    dry_run: bool = False,
    max_docs: int | None = None,
    deepread_max_pages: int | None = None,
    deepread_ocr_model: str | None = None,
    deepread_ocr_provider: str | None = None,
    mdocagent_max_pages: int | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    max_workers: int = 1,
) -> None:
    """Run the baseline sweep and write output_root/<dataset>/<experiment>/q<idx>/.

    Note: there is no `seed` parameter. None of the extractors use Python /
    NumPy global RNGs (temperature=0 LLM calls; deterministic retrieval), so a
    seed flag here would have no observable effect. `majority_vote_eval.py`
    has its own local `random.Random(seed)` for sample selection.
    """
    config = load_config(config_path)
    eval_provider = eval_provider or llm_provider
    eval_model = eval_model or llm_model

    extractor = _get_extractor(
        experiment,
        deepread_max_pages=deepread_max_pages,
        deepread_ocr_model=deepread_ocr_model,
        deepread_ocr_provider=deepread_ocr_provider,
        mdocagent_max_pages=mdocagent_max_pages,
    )
    from agent.baselines.defaults import resolve_dataset_root

    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)
    dataset_root = str(resolve_dataset_root(config))
    dataset_key = _dataset_output_key(config)

    for query_idx in query_indices:
        query_text = get_query_text(dataset_root, query_idx)
        doc_ids = _query_doc_ids(config, query_idx)
        if not doc_ids:
            print(f"[baseline-{experiment}] q{query_idx}: no docs configured, skipping")
            continue

        if max_docs is not None:
            doc_ids = doc_ids[:max_docs]
            logger.info("Capped doc set to %d docs (--max-docs)", max_docs)

        out_dir = output_root / dataset_key / experiment / f"q{query_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[baseline-{experiment}] q{query_idx}: {len(doc_ids)} docs, "
            f"provider={llm_provider}/{llm_model}"
        )

        if dry_run:
            print(f"  [dry-run] would process: {doc_ids}")
            continue

        def _process_cell(doc_id: str) -> DeployedRow:
            wrapper_t0 = time.perf_counter()
            result_latency_ms: float | None = None
            try:
                doc_inputs: DocInputs = build_doc_inputs(config, query_idx, doc_id)
                extract_kwargs: dict[str, Any] = {
                    "query_idx": query_idx,
                    "query_text": query_text,
                    "doc_id": doc_id,
                    "doc_inputs": doc_inputs,
                    "cached_caller": cached_caller,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
                if experiment == "qa-agent":
                    extract_kwargs["embedding_provider"] = embedding_provider
                    extract_kwargs["embedding_model"] = embedding_model
                result = extractor.extract(**extract_kwargs)
                result_latency_ms = result.latency_ms
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

            # Prefer the extractor's own latency_ms when it succeeded so the
            # console line matches what landed in the JSONL row; fall back to
            # wrapper-measured elapsed when the extractor errored before
            # producing a result.
            latency_ms = (
                result_latency_ms
                if result_latency_ms is not None
                else (time.perf_counter() - wrapper_t0) * 1000.0
            )
            status = "pass" if row["judge_result"] is True else "fail"
            gen_cost = float(row.get("gen_cost_usd", 0.0) or 0.0)
            judge_cost = float(row.get("judge_cost_usd", 0.0) or 0.0)
            print(
                f"  {doc_id}: {status} "
                f"gen=${gen_cost:.4f} judge=${judge_cost:.4f} "
                f"cost=${row['actual_cost_usd']:.4f} "
                f"latency={latency_ms:.0f}ms"
            )
            return row

        rows: list[DeployedRow] = []
        if max_workers <= 1:
            for doc_id in doc_ids:
                rows.append(_process_cell(doc_id))
        else:
            # Parallelise the (query, doc) cells of one query. Different cells
            # write to independent run_name dirs in upstream MDocAgent; the
            # only shared path is the prepare_inputs critical section, which
            # is guarded by a lock in extractor.py.
            n_workers = min(max_workers, len(doc_ids))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_cell, d): d for d in doc_ids}
                for future in as_completed(futures):
                    rows.append(future.result())

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
