"""Self-consistency ground-truth pipeline.

The pipeline keeps candidate generation artifacts separate from the final
answer-only ground truth used by downstream court/baseline runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from core.pipeline.e2e_utils.cache import CacheResult, DEFAULT_CACHE_DB_PATH

import gt_gen.generator as generator
import gt_gen.judge as judge


@dataclass(frozen=True)
class CandidateRun:
    """One all-mode candidate generation run."""

    index: int
    root: Path
    temperature: float
    summary: generator.BatchGenerationSummary


DEFAULT_ALL_RUNS = 3
DEFAULT_SELF_CONSISTENCY_TEMPERATURE = 0.2
DEFAULT_JUDGE_MODE: judge.JudgeMode = "batched"


def run_all_self_consistency_pipeline(
    *,
    target_dir: str | Path,
    work_dir: str | Path,
    query_indices: Sequence[int] | None = None,
    num_doc: int | None = None,
    seed: int = 42,
    llm_provider: str = generator.DEFAULT_LLM_PROVIDER,
    model: str = "gpt-5.4",
    input_mode: generator.InputMode = generator.DEFAULT_INPUT_MODE,
    all_runs: int = DEFAULT_ALL_RUNS,
    self_consistency_temperature: float = DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
    deterministic_first: bool = True,
    max_tokens: int = 2400,
    judge_max_tokens: int | None = None,
    judge_mode: judge.JudgeMode = DEFAULT_JUDGE_MODE,
    cache_db: str | Path | None = None,
    progress_cost: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all-mode self-consistency generation and judge disagreements.

    The final dataset is written under ``work_dir/final/<dataset>/latest``. Its
    ``ground_truth`` directory contains only answer values, so downstream methods
    do not see GT-generation hints, candidates, or judge rationale.
    """
    if all_runs < 1:
        raise ValueError("all_runs must be >= 1")
    resolved_provider = generator.normalize_llm_provider(llm_provider)
    resolved_model = generator.resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = generator.resolve_input_mode(resolved_provider, input_mode)
    source_root = generator.resolve_dataset_root(target_dir)
    dataset_name = _dataset_name(source_root)
    queries = generator.load_queries(source_root / "queries.json", query_indices)
    selected_pdfs = generator.sample_pdf_paths(
        source_root / "raw", num_doc=num_doc, seed=seed
    )
    root_work_dir = Path(work_dir).expanduser().resolve()
    _prepare_work_dir(root_work_dir, overwrite=overwrite)

    run_t0 = time.perf_counter()
    final_root = root_work_dir / "final" / dataset_name / "latest"
    _copy_dataset_subset(source_root, final_root, selected_pdfs)

    candidate_runs: list[CandidateRun] = []
    temperatures = _candidate_temperatures(
        all_runs=all_runs,
        deterministic_first=deterministic_first,
        self_consistency_temperature=self_consistency_temperature,
    )
    base_cache_db = Path(cache_db).expanduser() if cache_db is not None else None
    for run_index, temperature in enumerate(temperatures):
        run_root = root_work_dir / f"all_run_{run_index}" / dataset_name / "latest"
        _copy_dataset_subset(source_root, run_root, selected_pdfs)
        run_cache_db = (
            str(base_cache_db)
            if base_cache_db is not None
            else str(root_work_dir / f"all_run_{run_index}" / "cache.db")
        )
        run_summary = generator.generate_ground_truth_for_queries(
            target_dir=run_root,
            query_indices=[q.idx for q in queries],
            num_doc=None,
            llm_provider=resolved_provider,
            model=resolved_model,
            seed=seed,
            input_mode=resolved_input_mode,
            generation_mode="all",
            cache_db=run_cache_db,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_cost=progress_cost,
            log_dir=root_work_dir / f"all_run_{run_index}" / "logs",
        )
        candidate_runs.append(
            CandidateRun(
                index=run_index,
                root=run_root,
                temperature=temperature,
                summary=run_summary,
            )
        )

    full_manifest, judge_manifest, decisions = _build_manifests_and_consensus(
        final_root=final_root,
        queries=queries,
        selected_pdfs=selected_pdfs,
        candidate_runs=candidate_runs,
        model=resolved_model,
    )
    candidate_manifest_path = root_work_dir / "candidates_manifest.json"
    judge_manifest_path = root_work_dir / "judge_manifest.json"
    _write_json(candidate_manifest_path, full_manifest)
    _write_json(judge_manifest_path, judge_manifest)

    judge_cache_db = (
        str(base_cache_db)
        if base_cache_db is not None
        else str(root_work_dir / "judge" / "cache.db")
    )
    judge_call_rows, judge_failed_count = _judge_disagreements(
        final_root=final_root,
        queries=queries,
        judge_manifest=judge_manifest,
        decisions=decisions,
        llm_provider=resolved_provider,
        model=resolved_model,
        input_mode=resolved_input_mode,
        judge_mode=judge_mode,
        cache_db=judge_cache_db,
        max_tokens=judge_max_tokens if judge_max_tokens is not None else max_tokens,
        progress_cost=progress_cost,
    )
    _write_decision_files(final_root, decisions)

    generation_metrics = _generation_metrics(candidate_runs)
    judge_metrics = _judge_metrics(judge_call_rows)
    cell_count = len(selected_pdfs) * len(queries)
    summary_payload = {
        "pipeline": "all_self_consistency_judge_on_disagreement",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset_root": str(source_root),
        "final_dataset_root": str(final_root),
        "work_dir": str(root_work_dir),
        "candidate_manifest_path": str(candidate_manifest_path),
        "judge_manifest_path": str(judge_manifest_path),
        "dataset": dataset_name,
        "model": resolved_model,
        "llm_provider": resolved_provider,
        "input_mode": resolved_input_mode,
        "generation_mode": "all",
        "all_runs": all_runs,
        "temperatures": temperatures,
        "judge_model": resolved_model,
        "judge_mode": judge_mode,
        "max_tokens": max_tokens,
        "judge_max_tokens": judge_max_tokens if judge_max_tokens is not None else max_tokens,
        "seed": seed,
        "num_doc": len(selected_pdfs),
        "query_indices": [q.idx for q in queries],
        "doc_ids": [p.stem for p in selected_pdfs],
        "cell_count": cell_count,
        "candidate_answer_count": _candidate_answer_count(full_manifest),
        "accepted_without_judge_count": _count_decisions(decisions, "consensus"),
        "disagreement_cell_count": _count_decisions(decisions, "judge_pending")
        + _count_decisions(decisions, "judged"),
        "judged_count": _count_decisions(decisions, "judged"),
        "missing_candidate_cell_count": _count_decisions(decisions, "missing_candidates"),
        "generation_failed_count": sum(
            run.summary.failed_count for run in candidate_runs
        ),
        "judge_failed_count": judge_failed_count,
        "failed_count": sum(run.summary.failed_count for run in candidate_runs)
        + judge_failed_count
        + _count_decisions(decisions, "missing_candidates"),
        "generation_metrics": generation_metrics,
        "judge_metrics": judge_metrics,
        "total_cost_usd": round(
            generation_metrics["cost_usd"] + judge_metrics["cost_usd"], 6
        ),
        "run_latency_ms": round((time.perf_counter() - run_t0) * 1000.0, 3),
        "candidate_runs": [
            {
                "index": run.index,
                "root": str(run.root),
                "temperature": run.temperature,
                "log_path": str(run.summary.log_path) if run.summary.log_path else None,
                "generated_count": run.summary.generated_count,
                "failed_count": run.summary.failed_count,
                "skipped_existing_count": run.summary.skipped_existing_count,
                "run_latency_ms": round(run.summary.run_latency_ms, 3),
            }
            for run in candidate_runs
        ],
        "judge_calls": judge_call_rows,
    }
    summary_path = root_work_dir / "pipeline_summary.json"
    summary_payload["summary_path"] = str(summary_path)
    _write_json(summary_path, summary_payload)
    return summary_payload


def _candidate_temperatures(
    *,
    all_runs: int,
    deterministic_first: bool,
    self_consistency_temperature: float,
) -> list[float]:
    if deterministic_first:
        return [0.0] + [float(self_consistency_temperature)] * (all_runs - 1)
    return [float(self_consistency_temperature)] * all_runs


def _prepare_work_dir(work_dir: Path, *, overwrite: bool) -> None:
    if work_dir.exists() and any(work_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{work_dir} is not empty; pass overwrite=True")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def _copy_dataset_subset(
    source_root: Path, dest_root: Path, selected_pdfs: Sequence[Path]
) -> None:
    raw_dir = dest_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "queries.json", dest_root / "queries.json")
    for pdf_path in selected_pdfs:
        shutil.copy2(pdf_path, raw_dir / pdf_path.name)


def _build_manifests_and_consensus(
    *,
    final_root: Path,
    queries: Sequence[generator.QuerySpec],
    selected_pdfs: Sequence[Path],
    candidate_runs: Sequence[CandidateRun],
    model: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    full_docs: list[dict[str, Any]] = []
    judge_docs: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, dict[str, Any]]] = {}
    dataset = _dataset_name(final_root)

    for source_pdf in selected_pdfs:
        doc_id = source_pdf.stem
        final_pdf_path = final_root / "raw" / source_pdf.name
        doc_candidates: dict[str, list[dict[str, Any]]] = {}
        judge_candidates: dict[str, list[dict[str, Any]]] = {}
        decisions[doc_id] = {}
        for query in queries:
            candidates = _load_candidates_for_cell(
                candidate_runs=candidate_runs,
                doc_id=doc_id,
                query_idx=query.idx,
                model=model,
            )
            doc_candidates[str(query.idx)] = candidates
            unique_candidates = _unique_candidates(candidates)
            if not unique_candidates:
                decisions[doc_id][str(query.idx)] = {
                    "decision": "missing_candidates",
                    "answer": None,
                    "candidates": [],
                    "reasoning": "No candidate answer was generated.",
                }
                continue
            if len(unique_candidates) == 1:
                answer = unique_candidates[0]["answer"]
                _merge_answer(final_root, doc_id, query.idx, answer)
                decisions[doc_id][str(query.idx)] = {
                    "decision": "consensus",
                    "answer": answer,
                    "picked_index": 0,
                    "picked_source": unique_candidates[0]["source"],
                    "candidates": unique_candidates,
                    "reasoning": (
                        f"{len(candidates)} all-mode candidate runs normalized "
                        "to the same answer."
                    ),
                }
            else:
                judge_candidates[str(query.idx)] = unique_candidates
                decisions[doc_id][str(query.idx)] = {
                    "decision": "judge_pending",
                    "answer": None,
                    "candidates": unique_candidates,
                    "reasoning": "All-mode self-consistency candidates disagreed.",
                }

        full_docs.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(final_pdf_path),
                "candidates": doc_candidates,
            }
        )
        if judge_candidates:
            judge_docs.append(
                {
                    "doc_id": doc_id,
                    "pdf_path": str(final_pdf_path),
                    "candidates": judge_candidates,
                }
            )

    full_manifest = {
        "dataset": dataset,
        "mode": "all_self_consistency",
        "documents": full_docs,
    }
    judge_manifest = {
        "dataset": dataset,
        "mode": "judge_disagreements_only",
        "documents": judge_docs,
    }
    return full_manifest, judge_manifest, decisions


def _load_candidates_for_cell(
    *,
    candidate_runs: Sequence[CandidateRun],
    doc_id: str,
    query_idx: int,
    model: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run in candidate_runs:
        output_path = run.root / "ground_truth" / f"{doc_id}.txt_answers.json"
        if not output_path.exists():
            continue
        data = json.loads(output_path.read_text(encoding="utf-8"))
        key = str(query_idx)
        if not isinstance(data, dict) or key not in data:
            continue
        source = f"all_run_{run.index}@{model}:t{run.temperature:g}"
        candidates.append({"source": source, "answer": data[key]})
    return candidates


def _unique_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for candidate in candidates:
        key = normalized_answer_key(candidate.get("answer"))
        if key not in by_key:
            by_key[key] = {
                "source": candidate["source"],
                "answer": candidate.get("answer"),
                "source_count": 1,
                "sources": [candidate["source"]],
            }
        else:
            by_key[key]["source_count"] += 1
            by_key[key]["sources"].append(candidate["source"])
            by_key[key]["source"] = "|".join(by_key[key]["sources"])
    return list(by_key.values())


def normalized_answer_key(answer: Any) -> str:
    """Canonical key for self-consistency equality.

    Lists are treated as order-insensitive because GT prompts ask for document
    order, but candidate order flips should not force a judge call by itself.
    """
    return json.dumps(
        _normalize_answer_for_key(answer),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_answer_for_key(answer: Any) -> Any:
    if isinstance(answer, str):
        return " ".join(answer.casefold().split())
    if isinstance(answer, list):
        return sorted(_normalize_answer_for_key(item) for item in answer)
    if isinstance(answer, dict):
        return {
            str(key): _normalize_answer_for_key(value)
            for key, value in sorted(answer.items(), key=lambda item: str(item[0]))
        }
    return answer


def _judge_disagreements(
    *,
    final_root: Path,
    queries: Sequence[generator.QuerySpec],
    judge_manifest: dict[str, Any],
    decisions: dict[str, dict[str, dict[str, Any]]],
    llm_provider: str,
    model: str,
    input_mode: generator.ResolvedInputMode,
    judge_mode: judge.JudgeMode,
    cache_db: str,
    max_tokens: int,
    progress_cost: bool,
) -> tuple[list[dict[str, Any]], int]:
    query_by_idx = {query.idx: query for query in queries}
    call_rows: list[dict[str, Any]] = []
    failed_count = 0
    for doc_entry in judge_manifest["documents"]:
        doc_id = doc_entry["doc_id"]
        pdf_path = Path(doc_entry["pdf_path"])
        candidates_by_query = {
            int(qidx): [
                judge.Candidate(
                    source=str(candidate["source"]),
                    answer=candidate.get("answer"),
                    reasoning=str(candidate.get("reasoning", "") or ""),
                )
                for candidate in candidates
            ]
            for qidx, candidates in doc_entry["candidates"].items()
        }
        pending_queries = [query_by_idx[qidx] for qidx in sorted(candidates_by_query)]
        try:
            judged_answers, cache_results = judge.judge_document(
                pdf_path=pdf_path,
                queries=pending_queries,
                candidates_by_query=candidates_by_query,
                judge_mode=judge_mode,
                llm_provider=llm_provider,
                model=model,
                input_mode=input_mode,
                cache_db=cache_db,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            failed_count += len(pending_queries)
            for query in pending_queries:
                decisions[doc_id][str(query.idx)]["decision"] = "judge_failed"
                decisions[doc_id][str(query.idx)]["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            continue

        for judged in judged_answers:
            _merge_answer(final_root, doc_id, judged.query_idx, judged.answer)
            decisions[doc_id][str(judged.query_idx)].update(
                {
                    "decision": "judged",
                    "answer": judged.answer,
                    "picked_index": judged.picked_index,
                    "picked_source": judged.picked_source,
                    "reasoning": judged.reasoning,
                }
            )

        if judge_mode == "batched":
            cache_result = cache_results[0]
            call_rows.append(
                _judge_call_row(
                    doc_id=doc_id,
                    query_indices=[q.idx for q in pending_queries],
                    cache_result=cache_result,
                )
            )
        else:
            for query, cache_result in zip(pending_queries, cache_results):
                call_rows.append(
                    _judge_call_row(
                        doc_id=doc_id,
                        query_indices=[query.idx],
                        cache_result=cache_result,
                    )
                )
        if progress_cost:
            latest = call_rows[-1]
            print(
                "[pipeline:judge] "
                f"doc={doc_id} queries={latest['query_indices']} "
                f"lat={latest['latency_ms']:.1f}ms cost=${latest['cost_usd']:.6f}"
            )
    return call_rows, failed_count


def _judge_call_row(
    *,
    doc_id: str,
    query_indices: Sequence[int],
    cache_result: CacheResult,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "query_indices": list(query_indices),
        "cache_hit": cache_result.cache_hit,
        "input_tokens": cache_result.input_tokens,
        "cached_input_tokens": cache_result.cached_input_tokens,
        "output_tokens": cache_result.output_tokens,
        "cost_usd": round(cache_result.cost_usd, 6),
        "latency_ms": round(cache_result.latency_ms, 3),
    }


def _write_decision_files(
    final_root: Path, decisions: dict[str, dict[str, dict[str, Any]]]
) -> None:
    judged_dir = final_root / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, doc_decisions in decisions.items():
        _write_json(judged_dir / f"{doc_id}.judged_answers.json", doc_decisions)


def _merge_answer(final_root: Path, doc_id: str, query_idx: int, answer: Any) -> None:
    output_path = final_root / "ground_truth" / f"{doc_id}.txt_answers.json"
    data: dict[str, Any] = {}
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{output_path} must contain a JSON object")
        data = loaded
    data[str(query_idx)] = answer
    _write_json(output_path, data)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    generator._atomic_write_json(path, data)


def _generation_metrics(candidate_runs: Sequence[CandidateRun]) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "local_cache_hits": 0,
        "api_latency_ms": 0.0,
    }
    for run in candidate_runs:
        metrics = generator._run_metrics(run.summary.results)
        totals["input_tokens"] += metrics.input_tokens
        totals["cached_input_tokens"] += metrics.cached_input_tokens
        totals["output_tokens"] += metrics.output_tokens
        totals["cost_usd"] += metrics.cost_usd
        totals["local_cache_hits"] += metrics.local_cache_hits
        totals["api_latency_ms"] += metrics.api_latency_ms
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["api_latency_ms"] = round(totals["api_latency_ms"], 3)
    return totals


def _judge_metrics(call_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_tokens": sum(int(row["input_tokens"]) for row in call_rows),
        "cached_input_tokens": sum(
            int(row["cached_input_tokens"]) for row in call_rows
        ),
        "output_tokens": sum(int(row["output_tokens"]) for row in call_rows),
        "cost_usd": round(sum(float(row["cost_usd"]) for row in call_rows), 6),
        "local_cache_hits": sum(1 for row in call_rows if row["cache_hit"]),
        "api_latency_ms": round(
            sum(float(row["latency_ms"]) for row in call_rows if not row["cache_hit"]),
            3,
        ),
    }


def _candidate_answer_count(manifest: dict[str, Any]) -> int:
    return sum(
        len(candidates)
        for doc in manifest["documents"]
        for candidates in doc["candidates"].values()
    )


def _count_decisions(
    decisions: dict[str, dict[str, dict[str, Any]]], decision: str
) -> int:
    return sum(
        1
        for doc_decisions in decisions.values()
        for row in doc_decisions.values()
        if row.get("decision") == decision
    )


def _dataset_name(dataset_root: Path) -> str:
    return (
        dataset_root.parent.name
        if dataset_root.name == "latest"
        else dataset_root.name
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GT via all-mode self-consistency and judge only "
            "disagreements."
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--query-indices",
        default="all",
        help="Comma-separated 1-based query indices, ranges like 1-5, or 'all'.",
    )
    parser.add_argument("--num-doc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm-provider",
        choices=["azure", "claude-code", "claude"],
        default=generator.DEFAULT_LLM_PROVIDER,
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "text", "claude-read-pdf"],
        default=generator.DEFAULT_INPUT_MODE,
    )
    parser.add_argument("--all-runs", type=int, default=DEFAULT_ALL_RUNS)
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
    )
    parser.add_argument(
        "--no-deterministic-first",
        action="store_true",
        help="Use self-consistency temperature for every all-mode run.",
    )
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--judge-mode",
        choices=["per-query", "batched"],
        default=DEFAULT_JUDGE_MODE,
    )
    parser.add_argument("--cache-db", default=None)
    parser.add_argument("--progress-cost", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    query_indices = generator._parse_query_indices_arg(args.query_indices)
    summary = run_all_self_consistency_pipeline(
        target_dir=args.target_dir,
        work_dir=args.work_dir,
        query_indices=query_indices,
        num_doc=args.num_doc,
        seed=args.seed,
        llm_provider=args.llm_provider,
        model=args.model,
        input_mode=args.input_mode,
        all_runs=args.all_runs,
        self_consistency_temperature=args.self_consistency_temperature,
        deterministic_first=not args.no_deterministic_first,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        judge_mode=args.judge_mode,
        cache_db=args.cache_db,
        progress_cost=args.progress_cost,
        overwrite=args.overwrite,
    )
    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "----- GT Pipeline Summary -----\n"
        f"Final Root:     {summary['final_dataset_root']}\n"
        f"Docs:           {summary['num_doc']}\n"
        f"Queries:        {len(summary['query_indices'])}\n"
        f"Cells:          {summary['cell_count']}\n"
        f"Consensus:      {summary['accepted_without_judge_count']}\n"
        f"Disagreements:  {summary['disagreement_cell_count']}\n"
        f"Judged:         {summary['judged_count']}\n"
        f"Failed:         {summary['failed_count']}\n"
        f"Gen Cost:       ${summary['generation_metrics']['cost_usd']:.6f}\n"
        f"Judge Cost:     ${summary['judge_metrics']['cost_usd']:.6f}\n"
        f"Total Cost:     ${summary['total_cost_usd']:.6f}\n"
        f"Summary:        {summary['summary_path']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
