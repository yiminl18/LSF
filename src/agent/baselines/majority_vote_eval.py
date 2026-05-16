"""Run unlabeled baseline sweeps and score by cross-baseline majority vote."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agent.baselines.base import BaselineExtractor, DocInputs
from agent.baselines.loader import build_doc_inputs
from agent.rule_runtime.data import get_query_text
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH

BASELINES: tuple[str, ...] = ("exit", "deepread", "mdocagent", "qa-agent")
DEFAULT_DATASETS: tuple[str, ...] = ("nopv", "court")
DEFAULT_DATA_ROOT = Path("datasets")
DEFAULT_OUTPUT_ROOT = Path("output/agent/baselines_majority_vote")
_INFO_NOT_FOUND = "information not found"
EquivalenceJudge = Callable[[str, str, str], bool]


@dataclass(frozen=True, slots=True)
class SamplePair:
    dataset: str
    dataset_root: str
    query_idx: int
    query_text: str
    doc_id: str


def _load_query_count(dataset_root: Path) -> int:
    queries_json = dataset_root / "queries.json"
    if queries_json.exists():
        data = json.loads(queries_json.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    queries_txt = dataset_root / "queries.txt"
    if queries_txt.exists():
        return len([line for line in queries_txt.read_text(encoding="utf-8").splitlines() if line.strip()])
    raise FileNotFoundError(f"No queries.json or queries.txt found under {dataset_root}")


def _raw_doc_ids(dataset_root: Path) -> list[str]:
    raw_dir = dataset_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    doc_ids = sorted(path.stem for path in raw_dir.glob("*.pdf"))
    if not doc_ids:
        raise FileNotFoundError(f"No PDFs found under {raw_dir}")
    return doc_ids


def sample_pairs(
    *,
    datasets: Sequence[str],
    data_root: Path,
    queries_per_dataset: int,
    docs_per_query: int,
    seed: int,
    query_indices: Sequence[int] | None = None,
) -> list[SamplePair]:
    rng = random.Random(seed)
    pairs: list[SamplePair] = []
    for dataset in datasets:
        dataset_root = data_root / dataset / "latest"
        query_count = _load_query_count(dataset_root)
        doc_ids = _raw_doc_ids(dataset_root)
        if query_indices is not None:
            bad = [idx for idx in query_indices if idx < 0 or idx >= query_count]
            if bad:
                raise ValueError(f"{dataset} query index out of range: {bad}")
            selected_query_indices = list(query_indices)
        else:
            if queries_per_dataset > query_count:
                raise ValueError(
                    f"{dataset} has only {query_count} queries; requested {queries_per_dataset}"
                )
            selected_query_indices = sorted(rng.sample(range(query_count), queries_per_dataset))
        if docs_per_query > len(doc_ids):
            raise ValueError(
                f"{dataset} has only {len(doc_ids)} docs; requested {docs_per_query}"
            )
        for query_idx in selected_query_indices:
            query_text = get_query_text(str(dataset_root), query_idx)
            selected_docs = sorted(rng.sample(doc_ids, docs_per_query))
            for doc_id in selected_docs:
                pairs.append(
                    SamplePair(
                        dataset=dataset,
                        dataset_root=str(dataset_root),
                        query_idx=query_idx,
                        query_text=query_text,
                        doc_id=doc_id,
                    )
                )
    return pairs


def _make_extractor(
    baseline: str,
    *,
    deepread_max_pages: int | None,
    max_doc_pages: int | None,
    deepread_ocr_provider: str | None,
    deepread_ocr_model: str | None,
) -> BaselineExtractor:
    if baseline == "exit":
        from agent.baselines.exit.extractor import ExitExtractor

        return ExitExtractor()
    if baseline == "deepread":
        from agent.baselines.deepread.extractor import DeepReadExtractor

        kwargs: dict[str, Any] = {}
        if deepread_max_pages is not None:
            kwargs["max_pages"] = deepread_max_pages
        if deepread_ocr_provider is not None:
            kwargs["ocr_provider"] = deepread_ocr_provider
        if deepread_ocr_model is not None:
            kwargs["ocr_model"] = deepread_ocr_model
        return DeepReadExtractor(**kwargs)
    if baseline == "mdocagent":
        from agent.baselines.mdocagent.extractor import MDocAgentExtractor

        return MDocAgentExtractor(max_pages=max_doc_pages)
    if baseline == "qa-agent":
        from agent.baselines.qa_agent.extractor import QAAgentExtractor

        return QAAgentExtractor()
    raise ValueError(f"Unknown baseline: {baseline!r}")


def _config_for_pair(pair: SamplePair) -> dict[str, Any]:
    return {
        "dataset": pair.dataset,
        "dataset_root": pair.dataset_root,
        "parser": "docling",
        "queries": [{"query_idx": pair.query_idx, "documents": [pair.doc_id]}],
    }


def _limit_doc_inputs(doc_inputs: DocInputs, max_doc_chars: int | None) -> DocInputs:
    if max_doc_chars is None or max_doc_chars <= 0:
        return doc_inputs
    if len(doc_inputs.normalized_text) <= max_doc_chars:
        return doc_inputs
    return DocInputs(
        normalized_text=doc_inputs.normalized_text[:max_doc_chars],
        entries=[],
        section_index={},
        pdf_path=doc_inputs.pdf_path,
        ground_truth=doc_inputs.ground_truth,
    )


def _limit_doc_pages(doc_inputs: DocInputs, max_doc_pages: int | None) -> DocInputs:
    if max_doc_pages is None or max_doc_pages <= 0 or not doc_inputs.pdf_path.exists():
        return doc_inputs
    import fitz

    pages: list[str] = []
    with fitz.open(str(doc_inputs.pdf_path)) as pdf:
        for page in pdf[:max_doc_pages]:
            pages.append(page.get_text())
    page_text = "\n\n".join(pages).strip()
    if not page_text:
        return doc_inputs
    return DocInputs(
        normalized_text=page_text,
        entries=[],
        section_index={},
        pdf_path=doc_inputs.pdf_path,
        ground_truth=doc_inputs.ground_truth,
    )


def _normalize_answer_for_vote(answer: str | None) -> str:
    if answer is None:
        return ""
    normalized = " ".join(str(answer).split()).strip().rstrip(".;:,")
    return normalized.casefold()


def _is_non_answer(normalized_answer: str) -> bool:
    return normalized_answer in {
        "",
        _INFO_NOT_FOUND,
        "not found",
        "unknown",
        "n/a",
        "none",
        "no answer",
        "cannot determine",
        "not available",
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _run_one(
    *,
    baseline: str,
    extractor: BaselineExtractor,
    pair: SamplePair,
    doc_inputs: DocInputs,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    include_trace: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        **asdict(pair),
        "baseline": baseline,
        "status": "ok",
        "answer": None,
        "normalized_answer": "",
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "error": None,
    }
    try:
        kwargs: dict[str, Any] = {
            "query_idx": pair.query_idx,
            "query_text": pair.query_text,
            "doc_id": pair.doc_id,
            "doc_inputs": doc_inputs,
            "cached_caller": cached_caller,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        }
        if baseline == "qa-agent":
            kwargs["embedding_provider"] = embedding_provider
            kwargs["embedding_model"] = embedding_model
        result = extractor.extract(**kwargs)
        row["answer"] = result.generated_answer
        row["normalized_answer"] = _normalize_answer_for_vote(result.generated_answer)
        row["cost_usd"] = result.cost_usd
        row["latency_ms"] = result.latency_ms
        if include_trace:
            row["trace"] = _jsonable(result.trace)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        row["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return row


def _cluster_equivalent_answers(
    rows: list[dict[str, Any]],
    *,
    equivalence_judge: EquivalenceJudge | None,
) -> list[list[dict[str, Any]]]:
    if equivalence_judge is None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["normalized_answer"]].append(row)
        return list(grouped.values())

    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        row_answer = str(row.get("answer") or "")
        matched = False
        for cluster in clusters:
            representative = str(cluster[0].get("answer") or "")
            if equivalence_judge(str(row.get("query_text") or ""), representative, row_answer):
                cluster.append(row)
                matched = True
                break
        if not matched:
            clusters.append([row])
    return clusters


def _majority_for_rows(
    rows: list[dict[str, Any]],
    *,
    equivalence_judge: EquivalenceJudge | None = None,
) -> dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("status") == "ok" and not _is_non_answer(row.get("normalized_answer", ""))
    ]
    if not valid:
        return {
            "majority_resolved": False,
            "majority_answer": None,
            "majority_normalized_answer": "",
            "majority_answer_variants": [],
            "majority_count": 0,
            "valid_vote_count": 0,
            "reason": "no_valid_answers",
        }

    clusters = _cluster_equivalent_answers(valid, equivalence_judge=equivalence_judge)
    clusters.sort(key=len, reverse=True)
    top_cluster = clusters[0]
    top_count = len(top_cluster)
    tied = len(clusters) > 1 and len(clusters[1]) == top_count
    needed = len(valid) // 2 + 1
    resolved = (not tied) and top_count >= needed
    display_answer = top_cluster[0].get("answer") if resolved else None
    top_norms = sorted({row["normalized_answer"] for row in top_cluster}) if resolved else []
    vote_counts = {
        " | ".join(sorted({row["normalized_answer"] for row in cluster})): len(cluster)
        for cluster in clusters
    }
    if resolved:
        for row in top_cluster:
            row["_majority_cluster_member"] = True
    return {
        "majority_resolved": resolved,
        "majority_answer": display_answer,
        "majority_normalized_answer": top_norms[0] if top_norms else "",
        "majority_answer_variants": top_norms,
        "majority_count": top_count,
        "valid_vote_count": len(valid),
        "reason": None if resolved else ("tie" if tied else "no_strict_majority"),
        "vote_counts": vote_counts,
    }


def apply_majority_vote(
    rows: list[dict[str, Any]],
    *,
    equivalence_judge: EquivalenceJudge | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["query_idx"]), row["doc_id"])].append(row)

    pair_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        dataset, query_idx, doc_id = key
        for row in group:
            row.pop("_majority_cluster_member", None)
        majority = _majority_for_rows(group, equivalence_judge=equivalence_judge)
        pair_row = {
            "dataset": dataset,
            "query_idx": query_idx,
            "doc_id": doc_id,
            "query_text": group[0].get("query_text", ""),
            **majority,
        }
        pair_rows.append(pair_row)
        for row in group:
            if majority["majority_resolved"] and row.get("status") == "ok":
                row["majority_vote_correct"] = bool(row.pop("_majority_cluster_member", False))
            else:
                row["majority_vote_correct"] = None
            row["majority_resolved"] = majority["majority_resolved"]
            row["majority_answer"] = majority["majority_answer"]
    return rows, pair_rows


def summarize(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pairs": len(pair_rows),
        "resolved_pairs": sum(1 for row in pair_rows if row["majority_resolved"]),
        "baselines": {},
        "datasets": {},
    }
    for baseline in sorted({row["baseline"] for row in rows}):
        b_rows = [row for row in rows if row["baseline"] == baseline]
        resolved = [row for row in b_rows if row.get("majority_vote_correct") is not None]
        summary["baselines"][baseline] = {
            "runs": len(b_rows),
            "ok": sum(1 for row in b_rows if row["status"] == "ok"),
            "errors": sum(1 for row in b_rows if row["status"] == "error"),
            "resolved": len(resolved),
            "majority_vote_correct": sum(1 for row in resolved if row["majority_vote_correct"]),
            "majority_vote_accuracy": (
                sum(1 for row in resolved if row["majority_vote_correct"]) / len(resolved)
                if resolved else None
            ),
            "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in b_rows), 6),
        }
    for dataset in sorted({row["dataset"] for row in pair_rows}):
        d_pairs = [row for row in pair_rows if row["dataset"] == dataset]
        summary["datasets"][dataset] = {
            "pairs": len(d_pairs),
            "resolved_pairs": sum(1 for row in d_pairs if row["majority_resolved"]),
        }
    return summary


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _build_llm_equivalence_judge(
    cached_caller: CachedLLMCaller,
    *,
    llm_provider: str,
    llm_model: str,
) -> EquivalenceJudge:
    schema = {
        "name": "answer_equivalence",
        "description": "Whether two candidate answers are equivalent for a question.",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "equivalent": {"type": "boolean"},
            },
            "required": ["equivalent"],
        },
    }

    def judge(question: str, answer_a: str, answer_b: str) -> bool:
        if _normalize_answer_for_vote(answer_a) == _normalize_answer_for_vote(answer_b):
            return True
        prompt = (
            "Decide whether two candidate answers express the same final answer "
            "for the question. Minor wording, punctuation, and added explanatory "
            "phrases do not matter. Different values or missing values are not equivalent.\n\n"
            f"Question: {question}\n"
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}\n\n"
            "Return JSON only."
        )
        result = cached_caller.call(
            prompt,
            llm_provider=llm_provider,
            model=llm_model,
            max_tokens=20,
            response_schema=schema,
        )
        try:
            parsed = json.loads(result.response)
        except json.JSONDecodeError:
            return result.response.strip().lower().startswith("yes")
        return bool(parsed.get("equivalent"))

    return judge


def run_majority_vote_eval(
    *,
    datasets: Sequence[str],
    data_root: Path,
    queries_per_dataset: int,
    docs_per_query: int,
    query_indices: Sequence[int] | None,
    baselines: Sequence[str],
    seed: int,
    output_dir: Path,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    deepread_max_pages: int | None,
    deepread_ocr_provider: str | None,
    deepread_ocr_model: str | None,
    max_doc_pages: int | None,
    max_doc_chars: int | None,
    llm_vote: bool,
    dry_run: bool,
    include_trace: bool,
) -> dict[str, Any]:
    pairs = sample_pairs(
        datasets=datasets,
        data_root=data_root,
        queries_per_dataset=queries_per_dataset,
        docs_per_query=docs_per_query,
        seed=seed,
        query_indices=query_indices,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample_plan.json"
    sample_path.write_text(
        json.dumps([asdict(pair) for pair in pairs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if dry_run:
        print(f"[majority-vote] sample -> {sample_path}")
        for dataset in datasets:
            n = sum(1 for pair in pairs if pair.dataset == dataset)
            print(f"  {dataset}: {n} pairs")
        return {"dry_run": True, "sample_path": str(sample_path), "pairs": len(pairs)}

    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)
    extractors = {
        baseline: _make_extractor(
            baseline,
            deepread_max_pages=deepread_max_pages,
            max_doc_pages=max_doc_pages,
            deepread_ocr_provider=deepread_ocr_provider,
            deepread_ocr_model=deepread_ocr_model,
        )
        for baseline in baselines
    }
    doc_cache: dict[tuple[str, str], DocInputs] = {}
    rows: list[dict[str, Any]] = []
    total = len(pairs) * len(baselines)
    done = 0

    for pair in pairs:
        doc_key = (pair.dataset, pair.doc_id)
        if doc_key not in doc_cache:
            doc_cache[doc_key] = _limit_doc_inputs(
                _limit_doc_pages(
                    build_doc_inputs(
                        _config_for_pair(pair),
                        pair.query_idx,
                        pair.doc_id,
                    ),
                    max_doc_pages,
                ),
                max_doc_chars,
            )
        doc_inputs = doc_cache[doc_key]
        for baseline in baselines:
            done += 1
            print(
                f"[majority-vote] {done}/{total} "
                f"{pair.dataset} q{pair.query_idx} {pair.doc_id} {baseline}",
                flush=True,
            )
            row = _run_one(
                baseline=baseline,
                extractor=extractors[baseline],
                pair=pair,
                doc_inputs=doc_inputs,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                include_trace=include_trace,
            )
            rows.append(row)

    equivalence_judge = (
        _build_llm_equivalence_judge(
            cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        if llm_vote
        else None
    )
    rows, pair_rows = apply_majority_vote(rows, equivalence_judge=equivalence_judge)
    summary = summarize(rows, pair_rows)
    summary.update(
        {
            "datasets_requested": list(datasets),
            "baselines_requested": list(baselines),
            "seed": seed,
            "queries_per_dataset": queries_per_dataset,
            "docs_per_query": docs_per_query,
            "query_indices": list(query_indices) if query_indices is not None else None,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "vote_method": "llm_equivalence" if llm_vote else "exact",
            "deepread_max_pages": deepread_max_pages,
            "max_doc_pages": max_doc_pages,
            "max_doc_chars": max_doc_chars,
        }
    )

    _write_jsonl(output_dir / "baseline_rows.jsonl", rows)
    _write_jsonl(output_dir / "pair_majority_rows.jsonl", pair_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[majority-vote] rows -> {output_dir / 'baseline_rows.jsonl'}")
    print(f"[majority-vote] pairs -> {output_dir / 'pair_majority_rows.jsonl'}")
    print(f"[majority-vote] summary -> {output_dir / 'summary.json'}")
    return summary


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unlabeled baselines with majority-vote scoring.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--queries-per-dataset", type=int, default=5)
    parser.add_argument(
        "--query-indices",
        default=None,
        help="Comma-separated query indices to use for every dataset, e.g. 0,1,2.",
    )
    parser.add_argument("--docs-per-query", type=int, default=5)
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--llm-provider", default="azure")
    parser.add_argument("--llm-model", default="gpt-5.4-mini")
    parser.add_argument("--embed-provider", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--deepread-max-pages", type=int, default=None)
    parser.add_argument(
        "--max-doc-pages",
        type=int,
        default=None,
        help="Optional cap on PDF pages converted into normalized_text for text baselines.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=None,
        help="Optional cap on normalized text passed to text-only baselines.",
    )
    parser.add_argument("--ocr-provider", default=None)
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--no-llm-vote", action="store_true")
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    datasets = _split_csv(args.datasets)
    baselines = _split_csv(args.baselines)
    query_indices = (
        [int(part) for part in _split_csv(args.query_indices)]
        if args.query_indices
        else None
    )
    unknown = sorted(set(baselines) - set(BASELINES))
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown)}")
    if args.queries_per_dataset <= 0 or args.docs_per_query <= 0:
        raise SystemExit("--queries-per-dataset and --docs-per-query must be positive")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    summary = run_majority_vote_eval(
        datasets=datasets,
        data_root=args.data_root,
        queries_per_dataset=args.queries_per_dataset,
        docs_per_query=args.docs_per_query,
        query_indices=query_indices,
        baselines=baselines,
        seed=args.seed,
        output_dir=output_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        embedding_provider=args.embed_provider,
        embedding_model=args.embed_model,
        deepread_max_pages=args.deepread_max_pages,
        deepread_ocr_provider=args.ocr_provider,
        deepread_ocr_model=args.ocr_model,
        max_doc_pages=args.max_doc_pages,
        max_doc_chars=args.max_doc_chars,
        llm_vote=not args.no_llm_vote,
        dry_run=args.dry_run,
        include_trace=args.include_trace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
