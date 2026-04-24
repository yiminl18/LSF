"""RAG baseline base class.

Defines the unified interface and shared gen+judge logic for all RAG retrieval baselines.
Subclasses need only implement preprocess_doc() and retrieve().
"""

from __future__ import annotations

import json
import math
import sys
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from core.embed.embeddings import get_model_name_for_provider, get_query_embedding

if TYPE_CHECKING:
    from core.pipeline.e2e import CostTracker
from core.pipeline.e2e_utils.cache import CachedLLMCaller, CacheResult
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.judge import judge_answer, normalize_ground_truth
from core.pipeline.e2e_utils.rag_vanilla import (
    build_context_with_budget,
    compute_ref_context_budget,
    generate_answer_from_text,
)


class BaseRAGBaseline(ABC):
    """Base class for all RAG baselines.

    Subclasses set name and implement preprocess_doc() and retrieve();
    shared gen+judge+budget logic is provided by process_doc_e2e().
    """

    name: str

    def __init__(
        self,
        processing_dir: Path,
        embed_provider: str,
        cached_caller: CachedLLMCaller,
        llm_provider: str,
        llm_model: str,
        cache_root: Path = Path(".cache"),
        chunk_size: int = 200,
        chunk_overlap: int = 0,
    ) -> None:
        self.processing_dir = processing_dir
        self.embed_provider = embed_provider
        self.embed_model = get_model_name_for_provider(embed_provider)
        self.cached_caller = cached_caller
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.cache_root = cache_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Accumulated preprocess cost (thread-safe).
        self._prep_lock = threading.Lock()
        self._prep_input_tokens = 0
        self._prep_output_tokens = 0
        self._prep_cost_usd = 0.0
        self._prep_latency_ms = 0.0
        self._prep_calls = 0
        self._prep_cache_hits = 0
        self._prep_docs_done = 0

    @abstractmethod
    def preprocess_doc(self, doc_id: str) -> None:
        """Preprocess a single document (results cached to disk)."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return a list of (passage_text, score) in descending relevance order."""
        ...

    def track_preprocess_call(self, result: CacheResult) -> None:
        """Record the LLM call cost for the preprocessing phase (thread-safe)."""
        cost = compute_cost(
            result.input_tokens,
            result.output_tokens,
            self.llm_provider,
            model=self.llm_model,
        )
        with self._prep_lock:
            self._prep_calls += 1
            self._prep_input_tokens += result.input_tokens
            self._prep_output_tokens += result.output_tokens
            self._prep_cost_usd += cost
            self._prep_latency_ms += result.latency_ms
            if result.cache_hit:
                self._prep_cache_hits += 1

    def get_preprocess_stats(self) -> dict:
        """Return a thread-safe snapshot of preprocessing cost statistics."""
        with self._prep_lock:
            calls = self._prep_calls
            return {
                "preprocess_calls": calls,
                "preprocess_input_tokens": self._prep_input_tokens,
                "preprocess_output_tokens": self._prep_output_tokens,
                "preprocess_cost_usd": self._prep_cost_usd,
                "preprocess_latency_ms": self._prep_latency_ms,
                "preprocess_cache_hits": self._prep_cache_hits,
                "preprocess_cache_hit_rate": (
                    self._prep_cache_hits / calls if calls > 0 else 0.0
                ),
            }

    def preprocess_all(self, doc_ids: list[str], workers: int = 1) -> None:
        """Preprocess all documents in batch (supports multi-threaded parallelism)."""
        total = len(doc_ids)
        sorted_ids = sorted(doc_ids)
        self._prep_docs_done = 0

        def _do_one(doc_id: str) -> None:
            try:
                self.preprocess_doc(doc_id)
            except Exception as e:
                print(
                    f"  ERROR preprocess {doc_id}: {type(e).__name__}: {str(e)[:200]}",
                    file=sys.stderr,
                    flush=True,
                )
            with self._prep_lock:
                self._prep_docs_done += 1
                done = self._prep_docs_done
            if done % 5 == 0 or done == total:
                stats = self.get_preprocess_stats()
                print(
                    f"  [{self.name}] preprocessed {done}/{total} | "
                    f"LLM calls={stats['preprocess_calls']} "
                    f"cost=${stats['preprocess_cost_usd']:.4f} "
                    f"cache={stats['preprocess_cache_hit_rate']:.0%}",
                    file=sys.stderr,
                    flush=True,
                )

        if workers <= 1:
            for doc_id in sorted_ids:
                _do_one(doc_id)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_do_one, doc_id): doc_id for doc_id in sorted_ids
                }
                for future in as_completed(futures):
                    doc_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(
                            f"  ERROR preprocess {doc_id}: {type(e).__name__}: {str(e)[:200]}",
                            file=sys.stderr,
                            flush=True,
                        )

        # Preprocessing complete — print summary.
        stats = self.get_preprocess_stats()
        if stats["preprocess_calls"] > 0:
            print(
                f"  [{self.name}] preprocess done: "
                f"{stats['preprocess_calls']} LLM calls, "
                f"${stats['preprocess_cost_usd']:.4f}, "
                f"cache={stats['preprocess_cache_hit_rate']:.0%}",
                file=sys.stderr,
                flush=True,
            )

    def get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding (reuses the project-wide global cache)."""
        return get_query_embedding(
            query, model=self.embed_model, provider=self.embed_provider
        )

    def process_doc_e2e(
        self,
        doc_id: str,
        q_idx: int,
        question: str,
        query_embedding: list[float],
        context_budget: int,
        gen_max_tokens: int,
        judge_max_tokens: int,
        gt_dir: Path,
        tracker: CostTracker,
    ) -> dict | None:
        """Shared gen+judge loop: retrieve → budget → gen → judge → result dict.

        Returns a standard doc result dict, or None when context is empty (skip).
        """
        # Dynamic top-k: retrieve enough chunks to fill the budget.
        needed_k = math.ceil(context_budget / max(self.chunk_size, 1)) + 5
        ranked = self.retrieve(question, query_embedding, doc_id, top_k=needed_k)

        # Budget truncation.
        context, chunks_used = build_context_with_budget(ranked, context_budget)

        if not context.strip():
            tracker.record_doc_done()
            return None

        # Generate answer.
        try:
            gen_result = generate_answer_from_text(
                question=question,
                context=context,
                cached_caller=self.cached_caller,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                max_tokens=gen_max_tokens,
            )
            tracker.record_gen(gen_result.cost_usd, gen_result.cache_hit)
        except Exception as e:
            print(
                f"  ERROR gen {doc_id}: {type(e).__name__}: {str(e)[:200]}",
                file=sys.stderr,
                flush=True,
            )
            tracker.record_doc_done()
            return {
                "document": doc_id,
                "gen_time_ms": 0.0,
                "judge_time_ms": 0.0,
                "gen_cost_usd": 0.0,
                "judge_cost_usd": 0.0,
                "accuracy": -2,
                "generated_answer": f"ERROR: {type(e).__name__}",
                "judge_result": "ERROR",
                "gen_cache_hit": False,
                "judge_cache_hit": False,
                "top_k_used": 0,
            }

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

            gt_key = str(q_idx + 1)
            if gt_key in gt_data:
                gt_normalized = normalize_ground_truth(gt_data[gt_key])
                try:
                    jr = judge_answer(
                        question=question,
                        generated_answer=gen_result.answer,
                        ground_truth=gt_normalized,
                        cached_caller=self.cached_caller,
                        llm_provider=self.llm_provider,
                        llm_model=self.llm_model,
                        max_tokens=judge_max_tokens,
                    )
                    accuracy = 1 if jr.is_correct else 0
                    judge_result_str = jr.raw_response
                    judge_time = jr.latency_ms
                    judge_cost = jr.cost_usd
                    judge_cache_hit = jr.cache_hit
                    tracker.record_judge(judge_cost, judge_cache_hit, jr.is_correct)
                except Exception as e:
                    print(
                        f"  ERROR judge {doc_id}: {type(e).__name__}: {str(e)[:200]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    accuracy = -2
                    judge_result_str = f"ERROR: {type(e).__name__}"

        tracker.record_doc_done()
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
            "top_k_used": chunks_used,
        }


def process_docs_baseline(
    baseline: BaseRAGBaseline,
    ref_detail: dict,
    ref_target_docs: list[dict],
    ref_eval_method: str,
    q_idx: int,
    question: str,
    query_embedding: list[float],
    gen_max_tokens: int,
    judge_max_tokens: int,
    gt_dir: Path,
    tracker: CostTracker,
) -> list[dict]:
    """Unified RAG baseline per-query processing loop.

    Iterates ref_target_docs, computes context budget, and calls baseline.process_doc_e2e().
    Replaces the original _process_docs_vanilla.
    """
    results = []
    for doc_entry in ref_target_docs:
        doc_id = doc_entry["doc_id"]

        context_budget = compute_ref_context_budget(
            ref_detail,
            doc_id,
            ref_eval_method,
        )
        if context_budget is None or context_budget <= 0:
            tracker.record_doc_done()
            continue

        result = baseline.process_doc_e2e(
            doc_id=doc_id,
            q_idx=q_idx,
            question=question,
            query_embedding=query_embedding,
            context_budget=context_budget,
            gen_max_tokens=gen_max_tokens,
            judge_max_tokens=judge_max_tokens,
            gt_dir=gt_dir,
            tracker=tracker,
        )
        if result is not None:
            results.append(result)

    return results
