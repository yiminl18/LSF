"""
End-to-end evaluation pipeline.

Flow: train → evaluate → generate answers → LLM judge → aggregate results.
Supports multi-threaded parallel gen+judge with thread-safe cost tracking and
real-time progress display.
"""

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from core.pipeline.e2e_utils.baselines import get_baseline, is_rag_baseline_config
from core.pipeline.e2e_utils.baselines.base import process_docs_baseline
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH
from core.pipeline.e2e_utils.generation import generate_answer
from core.pipeline.e2e_utils.judge import judge_answer, normalize_ground_truth
from core.pipeline.e2e_utils.results import build_query_result, build_summary
from core.pipeline.evaluate_model import evaluate_models
from core.pipeline.train_model import train_models
from core.llm.model import LLM_PROVIDERS
from core.utils.paths import PathManager


# ── Thread-safe cost / progress tracker ──────────────────────────────────


class CostTracker:
    """Thread-safe real-time cost / progress tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen_cost = 0.0
        self._judge_cost = 0.0
        self._gen_calls = 0
        self._judge_calls = 0
        self._cache_hits = 0
        self._total_calls = 0
        self._correct = 0
        self._judged = 0
        self._docs_done = 0
        self._docs_total = 0

    def set_total(self, total: int) -> None:
        with self._lock:
            self._docs_total = total

    def record_gen(self, cost: float, cache_hit: bool) -> None:
        with self._lock:
            self._gen_cost += cost
            self._gen_calls += 1
            self._total_calls += 1
            if cache_hit:
                self._cache_hits += 1

    def record_judge(self, cost: float, cache_hit: bool, is_correct: bool) -> None:
        with self._lock:
            self._judge_cost += cost
            self._judge_calls += 1
            self._total_calls += 1
            self._judged += 1
            if cache_hit:
                self._cache_hits += 1
            if is_correct:
                self._correct += 1

    def record_doc_done(self) -> None:
        with self._lock:
            self._docs_done += 1

    def snapshot(self) -> dict:
        """Return a thread-safe snapshot of current state."""
        with self._lock:
            total_cost = self._gen_cost + self._judge_cost
            acc = self._correct / self._judged if self._judged > 0 else 0.0
            hit_rate = (
                self._cache_hits / self._total_calls if self._total_calls > 0 else 0.0
            )
            return {
                "docs_done": self._docs_done,
                "docs_total": self._docs_total,
                "gen_cost": self._gen_cost,
                "judge_cost": self._judge_cost,
                "total_cost": total_cost,
                "accuracy": acc,
                "judged": self._judged,
                "correct": self._correct,
                "cache_hit_rate": hit_rate,
            }

    def progress_line(self, q_idx: int, doc_id: str) -> str:
        """Generate a single-line progress string."""
        s = self.snapshot()
        pct = s["docs_done"] / s["docs_total"] * 100 if s["docs_total"] else 0
        return (
            f"  [q{q_idx}] [{s['docs_done']}/{s['docs_total']} {pct:.0f}%] "
            f"{doc_id} | "
            f"cost=${s['total_cost']:.4f} (gen=${s['gen_cost']:.4f} judge=${s['judge_cost']:.4f}) | "
            f"acc={s['accuracy']:.2%} ({s['correct']}/{s['judged']}) | "
            f"cache={s['cache_hit_rate']:.0%}"
        )


# ── Utility functions ────────────────────────────────────────────────────


def _parse_seeds(seeds_str: str) -> list[int]:
    return [int(s.strip()) for s in seeds_str.split(",")]


def _load_eval_results(eval_results_dir: str, expected_dataset: str) -> dict:
    # Accept both eval output (results.json) and e2e output (summary.json).
    results_path = Path(eval_results_dir) / "results.json"
    if not results_path.exists():
        results_path = Path(eval_results_dir) / "summary.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found: {eval_results_dir}/\n"
            f"Expected: results.json or summary.json"
        )
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    actual_dataset = data.get("meta", {}).get("dataset", "")
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"Dataset mismatch: results have {actual_dataset!r}, "
            f"but --dataset is {expected_dataset!r}"
        )
    return data


def _log_phase_error(phase: str, doc_id: str, exc: Exception) -> None:
    """Log a phase-scoped LLM-call error to stderr (gen/judge)."""
    print(
        f"  ERROR {phase} {doc_id}: {type(exc).__name__}: {str(exc)[:200]}",
        file=sys.stderr,
        flush=True,
    )


def _gen_error_result(doc_id: str, top_k_used: int, exc: Exception) -> dict:
    """Build the sentinel per-doc result returned when generation fails.

    The accuracy=-2 sentinel signals "API error" to downstream aggregation
    (see results.build_query_result: rows with accuracy<0 are excluded).
    """
    return {
        "document": doc_id,
        "gen_time_ms": 0.0,
        "judge_time_ms": 0.0,
        "gen_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
        "accuracy": -2,
        "generated_answer": f"ERROR: {type(exc).__name__}",
        "judge_result": "ERROR",
        "gen_cache_hit": False,
        "judge_cache_hit": False,
        "top_k_used": top_k_used,
    }


def _process_doc(
    doc_detail: dict,
    q_idx: int,
    question: str,
    eval_method: str,
    top_k_min: int,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    gen_max_tokens: int,
    judge_max_tokens: int,
    gt_dir: Path,
    tracker: CostTracker,
) -> dict | None:
    """Process gen + judge for a single document (safe to call from a thread pool)."""
    doc_id = doc_detail["doc_id"]
    methods = doc_detail.get("methods", {})

    if eval_method not in methods:
        return None

    method_data = methods[eval_method]
    total_candidates = method_data["total_candidates"]
    top_5 = method_data["top_5"]

    effective_k = min(max(math.ceil(0.05 * total_candidates), top_k_min), 5)
    top_k_nodes = top_5[:effective_k]
    actual_k = len(top_k_nodes)

    # Generate answer (skip document on content filter / API error).
    try:
        gen_result = generate_answer(
            question=question,
            top_k_nodes=top_k_nodes,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_tokens=gen_max_tokens,
        )
    except Exception as e:
        _log_phase_error("gen", doc_id, e)
        tracker.record_doc_done()
        return _gen_error_result(doc_id, actual_k, e)
    tracker.record_gen(gen_result.cost_usd, gen_result.cache_hit)

    # Load GT and judge.
    accuracy = -1
    judge_result_str = "NO_GT"
    judge_time = 0.0
    judge_cost = 0.0
    judge_cache_hit = False

    gt_path = gt_dir / f"{doc_id}.txt_answers.json"
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_key = str(q_idx + 1)  # 1-indexed
        if gt_key in gt_data:
            gt_normalized = normalize_ground_truth(gt_data[gt_key])
            try:
                jr = judge_answer(
                    question=question,
                    generated_answer=gen_result.answer,
                    ground_truth=gt_normalized,
                    cached_caller=cached_caller,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    max_tokens=judge_max_tokens,
                )
                accuracy = 1 if jr.is_correct else 0
                judge_result_str = jr.raw_response
                judge_time = jr.latency_ms
                judge_cost = jr.cost_usd
                judge_cache_hit = jr.cache_hit
                tracker.record_judge(judge_cost, judge_cache_hit, jr.is_correct)
            except Exception as e:
                _log_phase_error("judge", doc_id, e)
                accuracy = -2
                judge_result_str = f"ERROR: {type(e).__name__}"

    tracker.record_doc_done()

    # Real-time progress (stderr to avoid mixing with JSON output).
    print(tracker.progress_line(q_idx, doc_id), file=sys.stderr, flush=True)

    return {
        "document": doc_id,
        "gen_time_ms": gen_result.latency_ms,
        "judge_time_ms": judge_time,
        "gen_cost_usd": gen_result.cost_usd,
        "judge_cost_usd": judge_cost,
        "accuracy": accuracy,
        "generated_answer": gen_result.answer,
        "judge_result": judge_result_str,
        "gen_cache_hit": gen_result.cache_hit,
        "judge_cache_hit": judge_cache_hit,
        "top_k_used": actual_k,
    }


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E2E Pipeline: train → eval → generate → judge → summary"
    )

    shared = parser.add_argument_group("shared (passthrough to train/eval)")
    shared.add_argument("--dataset", type=str, required=True, help="dataset name")
    shared.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        help="structural parser",
    )
    shared.add_argument(
        "--model-config", nargs="+", type=str, default=None, help="model type list"
    )
    shared.add_argument("--experiment", type=str, default="default", help="experiment ID")
    shared.add_argument(
        "--embed-provider", type=str, default="openrouter", help="embedding provider"
    )
    shared.add_argument(
        "--seeds", type=str, default="41,42,43", help="comma-separated seed list"
    )
    shared.add_argument("--limit", type=int, default=10, help="number of questions")
    shared.add_argument(
        "--questions",
        type=str,
        default=None,
        help="comma-separated question indices (0-based), e.g. 0,1,2,3,4",
    )
    shared.add_argument(
        "--workers", type=int, default=1, help="parallel workers for train/eval"
    )
    shared.add_argument("--score-agg", type=str, default="softmax", help="score aggregation method")
    shared.add_argument(
        "--softmax-alpha", type=float, default=5.0, help="softmax temperature parameter α"
    )
    shared.add_argument("--nn-device", type=str, default="auto", help="PyTorch device")
    shared.add_argument("--xgb-device", type=str, default="auto", help="XGBoost device")
    shared.add_argument(
        "--no-curriculum",
        action="store_true",
        help="disable curriculum learning (enabled by default)",
    )

    e2e = parser.add_argument_group("e2e-specific")
    e2e.add_argument(
        "--llm-provider",
        type=str,
        default="azure",
        choices=sorted(LLM_PROVIDERS),
        help="LLM provider (gen+judge)",
    )
    e2e.add_argument(
        "--model", type=str, required=True, help="LLM model (gen+judge)"
    )
    e2e.add_argument(
        "--top-k", type=int, default=1, help="minimum top-k (effective value computed by formula)"
    )
    e2e.add_argument("--eval-method", type=str, default=None, help="which method's ranking to use")
    e2e.add_argument(
        "--gen-max-tokens", type=int, default=500, help="max tokens for generation"
    )
    e2e.add_argument(
        "--judge-max-tokens", type=int, default=50, help="max tokens for judging"
    )
    e2e.add_argument("--skip-train", action="store_true", help="skip training")
    e2e.add_argument("--skip-eval", action="store_true", help="skip evaluation")
    e2e.add_argument(
        "--eval-results-dir", type=str, default=None, help="existing eval results directory"
    )
    e2e.add_argument(
        "--cache-db", type=str, default=DEFAULT_CACHE_DB_PATH, help="LLM cache DB path"
    )
    e2e.add_argument(
        "--e2e-workers",
        type=int,
        default=1,
        help="parallel threads for gen+judge (default 1 = sequential)",
    )

    # rag-baselines shared args
    vanilla = parser.add_argument_group("rag-baselines")
    vanilla.add_argument(
        "--ref-results-dir",
        type=str,
        default=None,
        help="xgb-v5 E2E results directory (token budget reference for rag-vanilla)",
    )
    vanilla.add_argument(
        "--chunk-size", type=int, default=200, help="chunk size (tokens, default=200)"
    )
    vanilla.add_argument(
        "--chunk-overlap", type=int, default=0, help="chunk overlap (tokens, default=0)"
    )

    return parser


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)

    model_configs = args.model_config if args.model_config else ["xgb-sem-struc-v5"]

    # Detect special baseline modes.
    is_rag_baseline = len(model_configs) == 1 and model_configs[0] == "rag-v1"
    # Detect all registered rag-* chunk-based baselines (rag-v1 excluded; it goes through the eval pipeline).
    is_rag_baseline_method = len(model_configs) == 1 and is_rag_baseline_config(
        model_configs[0]
    )

    if is_rag_baseline_method:
        # chunk-based baseline: skip train+eval, perform own retrieval, token budget from ref results.
        if not args.ref_results_dir:
            raise ValueError(
                f"--ref-results-dir is required for {model_configs[0]} (xgb-v5 E2E results)"
            )
        eval_model_configs = []
        eval_method = model_configs[0]
    elif is_rag_baseline:
        eval_model_configs: list[str] = []
        eval_method = "RAG"
    else:
        eval_model_configs = model_configs
        eval_method = args.eval_method if args.eval_method else model_configs[0]

    # Parse question indices: list passed to train/eval, set used for gen+judge filtering.
    questions_list: list[int] | None = None
    question_indices: set[int] | None = None
    if args.questions:
        questions_list = sorted(int(q.strip()) for q in args.questions.split(","))
        question_indices = set(questions_list)

    variant = args.parser if args.parser != "docling" else None
    paths = PathManager(experiment=args.experiment, processing_variant=variant)
    gt_dir = paths.get_ground_truth_dir(args.dataset)

    print("=== E2E Pipeline ===")
    print(f"Dataset:      {args.dataset}")
    print(f"Parser:       {args.parser}")
    print(f"Experiment:   {args.experiment}")
    print(f"Models:       {model_configs}")
    print(f"Eval Method:  {eval_method}")
    print(f"LLM Provider: {args.llm_provider}")
    print(f"LLM Model:    {args.model}")
    print(f"Top-k:        {args.top_k} (effective = min(max(ceil(5%*N), k), 5))")
    print(f"Seeds:        {seeds}")
    print(f"E2E Workers:  {args.e2e_workers}")
    print()

    # ── Phase 1: Train ──
    # train_timing: {q_idx: elapsed_ms}  (populated by train_models)
    train_timing: dict[int, float] = {}
    skip_train = args.skip_train or is_rag_baseline or is_rag_baseline_method
    if not skip_train:
        print("=== Phase 1: Training ===")
        t0 = time.perf_counter()
        train_timing = (
            train_models(
                dataset=args.dataset,
                limit=args.limit,
                model_configs=model_configs,
                experiment=args.experiment,
                provider=args.embed_provider,
                seeds=seeds,
                workers=args.workers,
                xgb_device=args.xgb_device,
                nn_device=args.nn_device,
                curriculum=not args.no_curriculum,
                parser=args.parser,
                questions=questions_list,
            )
            or {}
        )
        total_train = (time.perf_counter() - t0) * 1000.0
        print(f"\nTraining completed in {total_train:.0f}ms\n")
    else:
        reason = (
            model_configs[0]
            if is_rag_baseline_method
            else "rag-v1 baseline"
            if is_rag_baseline
            else "skipped"
        )
        print(f"=== Phase 1: Training ({reason}) ===\n")

    # ── Phase 2: Evaluate ──
    eval_timing: dict[int, float] = {}

    if is_rag_baseline_method:
        # chunk-based baseline: skip eval, load detail from ref E2E results (doc list + token budget).
        print(f"=== Phase 2: Evaluation ({model_configs[0]}, loading ref results) ===")
        ref_summary = _load_eval_results(args.ref_results_dir, args.dataset)
        ref_eval_method = ref_summary.get("meta", {}).get(
            "eval_method", "xgb-sem-struc-v5"
        )
        # Load per-query details from per_question list or details/ directory.
        ref_dir = Path(args.ref_results_dir)
        details = []
        for pq in ref_summary.get("per_question", []):
            qi = pq["q_idx"]
            detail_path = ref_dir / "details" / f"q{qi}.json"
            if detail_path.exists():
                with open(detail_path, encoding="utf-8") as f:
                    details.append(json.load(f))
        print(
            f"Loaded ref from: {args.ref_results_dir} ({len(details)} queries, method={ref_eval_method})\n"
        )
    elif not args.skip_eval:
        print("=== Phase 2: Evaluation ===")
        t0 = time.perf_counter()
        agg_path = evaluate_models(
            dataset=args.dataset,
            limit=args.limit,
            model_configs=eval_model_configs,
            experiment=args.experiment,
            provider=args.embed_provider,
            seeds=seeds,
            workers=args.workers,
            score_agg=args.score_agg,
            softmax_alpha=args.softmax_alpha,
            xgb_device=args.xgb_device,
            nn_device=args.nn_device,
            parser=args.parser,
            questions=questions_list,
        )
        total_eval = (time.perf_counter() - t0) * 1000.0
        print(f"\nEvaluation completed in {total_eval:.0f}ms\n")
        with open(agg_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        details = eval_data.get("details", [])
    else:
        print("=== Phase 2: Evaluation (skipped, loading existing) ===")
        if not args.eval_results_dir:
            raise ValueError("--eval-results-dir is required when using --skip-eval")
        eval_data = _load_eval_results(args.eval_results_dir, args.dataset)
        print(f"Loaded from: {args.eval_results_dir}\n")
        details = eval_data.get("details", [])

    # Extract per-question eval timing from eval results.
    for d in details:
        if "error" not in d and "_eval_elapsed_ms" in d:
            eval_timing[d["q_idx"]] = d["_eval_elapsed_ms"]

    # ── Phase 3+4: Generate + Judge ──
    print("=== Phase 3+4: Generate + Judge ===", file=sys.stderr, flush=True)
    cached_caller = CachedLLMCaller(db_path=args.cache_db)
    tracker = CostTracker()

    processing_dir = paths.get_processing_dir(args.dataset)

    # chunk-based baseline: instantiate and preprocess all documents.
    baseline = None
    if is_rag_baseline_method:
        baseline = get_baseline(
            model_configs[0],
            processing_dir=processing_dir,
            embed_provider=args.embed_provider,
            cached_caller=cached_caller,
            llm_provider=args.llm_provider,
            llm_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    # Pre-compute total document count (for progress percentage).
    total_docs_all = 0
    for detail in details:
        if "error" in detail:
            continue
        q_idx = detail.get("q_idx")
        if question_indices is not None and q_idx not in question_indices:
            continue
        if is_rag_baseline_method:
            total_docs_all += len(
                detail.get("eval_ranking_details", {}).get("target_docs_details", [])
            )
        else:
            total_docs_all += len(detail.get("target_docs_details", []))
    tracker.set_total(total_docs_all)

    # chunk-based baseline: pre-embed all documents (avoid per-doc API calls inside the query loop).
    if is_rag_baseline_method:
        all_doc_ids: list[str] = []
        seen: set[str] = set()
        for detail in details:
            if "error" in detail:
                continue
            q_idx = detail.get("q_idx")
            if question_indices is not None and q_idx not in question_indices:
                continue
            for d in detail.get("eval_ranking_details", {}).get(
                "target_docs_details", []
            ):
                if d["doc_id"] not in seen:
                    seen.add(d["doc_id"])
                    all_doc_ids.append(d["doc_id"])

        print(
            f"Pre-embedding {len(all_doc_ids)} docs "
            f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...",
            file=sys.stderr,
            flush=True,
        )
        baseline.preprocess_all(all_doc_ids, workers=args.e2e_workers)
        print("Pre-embedding done.\n", file=sys.stderr, flush=True)

    query_results: list[dict] = []
    skipped_count = 0

    for detail in details:
        if "error" in detail:
            q_idx_err = detail.get("q_idx", "?")
            print(
                f"  WARNING: q{q_idx_err} skipped (error: {detail['error']})",
                file=sys.stderr,
                flush=True,
            )
            skipped_count += 1
            continue

        q_idx = detail["q_idx"]
        if question_indices is not None and q_idx not in question_indices:
            continue

        question = detail["question"]

        if is_rag_baseline_method:
            # chunk-based baseline: doc list comes from ref detail.
            ref_target_docs = detail.get("eval_ranking_details", {}).get(
                "target_docs_details", []
            )
            query_embedding = baseline.get_query_embedding(question)
            doc_results = process_docs_baseline(
                baseline=baseline,
                ref_detail=detail,
                ref_target_docs=ref_target_docs,
                ref_eval_method=ref_eval_method,
                q_idx=q_idx,
                question=question,
                query_embedding=query_embedding,
                gen_max_tokens=args.gen_max_tokens,
                judge_max_tokens=args.judge_max_tokens,
                gt_dir=gt_dir,
                tracker=tracker,
            )
        else:
            target_docs = detail.get("target_docs_details", [])
            doc_results = _process_docs(
                target_docs=target_docs,
                q_idx=q_idx,
                question=question,
                eval_method=eval_method,
                args=args,
                cached_caller=cached_caller,
                gt_dir=gt_dir,
                tracker=tracker,
                max_workers=args.e2e_workers,
            )

        # Per-question cost summary.
        s = tracker.snapshot()
        print(
            f"\n  q{q_idx} done | running total: "
            f"acc={s['accuracy']:.2%} cost=${s['total_cost']:.4f} "
            f"cache={s['cache_hit_rate']:.0%}\n",
            file=sys.stderr,
            flush=True,
        )

        qr = build_query_result(
            q_idx=q_idx,
            question=question,
            doc_results=doc_results,
            eval_detail=detail if not is_rag_baseline_method else {},
            train_time_ms=train_timing.get(q_idx, 0.0),
            eval_time_ms=eval_timing.get(q_idx, 0.0),
        )
        query_results.append(qr)

    # ── Phase 5: Aggregate + Output ──
    print("\n=== Phase 5: Aggregate + Output ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_str = "+".join(model_configs) if model_configs else "default"

    results_dir = paths.get_results_dir(args.dataset)
    output_dir = (
        results_dir
        / f"e2e_{args.dataset}_{args.llm_provider}_k{args.top_k}_{model_str}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess cost (only present for RAG baselines).
    preprocess_stats = baseline.get_preprocess_stats() if is_rag_baseline_method else {}

    meta = {
        "dataset": args.dataset,
        "experiment": args.experiment,
        "model_configs": model_configs or [],
        "eval_method": eval_method,
        "parser": args.parser,
        "embed_provider": args.embed_provider,
        "llm_provider": args.llm_provider,
        "top_k_param": args.top_k,
        "top_k_formula": "min(max(ceil(5% * #headers), k), 5)",
        "e2e_workers": args.e2e_workers,
        "timestamp": timestamp,
        "training_time_ms": sum(train_timing.values()),
        "eval_time_ms": sum(eval_timing.values()),
        "num_questions_skipped": skipped_count,
        **(
            {
                "preprocess_cost_usd": preprocess_stats["preprocess_cost_usd"],
                "preprocess_latency_ms": preprocess_stats["preprocess_latency_ms"],
                "preprocess_llm_calls": preprocess_stats["preprocess_calls"],
                "preprocess_cache_hit_rate": preprocess_stats[
                    "preprocess_cache_hit_rate"
                ],
            }
            if preprocess_stats
            else {}
        ),
    }

    summary = build_summary(query_results, meta)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for qr in query_results:
        detail_path = details_dir / f"q{qr['q_idx']}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(qr, f, indent=2, ensure_ascii=False)

    # Print final summary.
    overall = summary["overall"]
    final = tracker.snapshot()
    print(f"\nResults: {output_dir}")
    print(f"Accuracy:    {overall['accuracy']:.4f}")
    print(
        f"Questions:   {overall['num_questions']} ({overall['num_questions_skipped']} skipped)"
    )
    print(f"Documents:   {overall['num_docs_total']}")
    print(f"Total Time:  {overall['total_time_ms']:.0f}ms")
    print(f"Total Cost:  ${overall['total_cost_usd']:.4f}")
    print(f"  Train:     {overall['training_time_ms']:.0f}ms / $0.00")
    print(f"  Eval:      {overall['eval_time_ms']:.0f}ms / $0.00")
    print(
        f"  Generate:  {overall['gen_time_ms']:.0f}ms / ${overall['gen_cost_usd']:.4f}"
    )
    print(
        f"  Judge:     {overall['judge_time_ms']:.0f}ms / ${overall['judge_cost_usd']:.4f}"
    )
    print(f"  Cache Hit: {final['cache_hit_rate']:.0%}")
    if preprocess_stats:
        prep_time_s = preprocess_stats["preprocess_latency_ms"] / 1000.0
        print(
            f"  Preproc:   {prep_time_s:.0f}s / ${preprocess_stats['preprocess_cost_usd']:.4f} "
            f"({preprocess_stats['preprocess_calls']} LLM calls, "
            f"cache={preprocess_stats['preprocess_cache_hit_rate']:.0%})"
        )


def _process_docs(
    target_docs: list[dict],
    q_idx: int,
    question: str,
    eval_method: str,
    args: argparse.Namespace,
    cached_caller: CachedLLMCaller,
    gt_dir: Path,
    tracker: CostTracker,
    max_workers: int,
) -> list[dict]:
    """Process all docs for one query; switch to a thread pool when max_workers > 1.

    Exception semantics differ by branch and are preserved verbatim from the
    prior split implementation:
    - Sequential: any exception from _process_doc propagates to the caller.
    - Parallel: exceptions from individual futures are caught and logged
      (gen+judge are I/O-bound; one bad doc shouldn't kill the query).
    """
    common_kwargs = dict(
        q_idx=q_idx,
        question=question,
        eval_method=eval_method,
        top_k_min=args.top_k,
        cached_caller=cached_caller,
        llm_provider=args.llm_provider,
        llm_model=args.model,
        gen_max_tokens=args.gen_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        gt_dir=gt_dir,
        tracker=tracker,
    )

    results: list[dict] = []
    if max_workers <= 1:
        for doc_detail in target_docs:
            r = _process_doc(doc_detail=doc_detail, **common_kwargs)
            if r is not None:
                results.append(r)
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_doc, doc_detail=doc_detail, **common_kwargs):
                doc_detail["doc_id"]
            for doc_detail in target_docs
        }
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"  ERROR: {doc_id}: {e}", file=sys.stderr, flush=True)
    return results


if __name__ == "__main__":
    main()
