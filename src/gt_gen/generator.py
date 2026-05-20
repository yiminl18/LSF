"""Ground-truth generator for one-query, one-document PDF evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from openai import OpenAI

from core.llm.cost import compute_cost_with_cached_input
from core.llm.tokens import estimate_tokens
from core.pipeline.e2e_utils.cache import (
    CacheResult,
    DEFAULT_CACHE_DB_PATH,
)

DEFAULT_LLM_PROVIDER = "azure"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_CLAUDE_MODEL = "sonnet"
DEFAULT_INPUT_MODE = "text"
DEFAULT_GENERATION_MODE = "single"
DEFAULT_LOG_DIR = "logs/gt_gen"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_CLAUDE_TIMEOUT_SEC = 600
GT_PROMPT_DIR = Path(__file__).with_name("prompts")
DEFAULT_GT_PROMPT_NAME = "default.txt"

InputMode = Literal["auto", "text", "claude-read-pdf"]
ResolvedInputMode = Literal["text", "claude-read-pdf"]
GenerationMode = Literal["single", "all"]

_CACHE_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    TEXT PRIMARY KEY,
    prompt_text  TEXT,
    response     TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    latency_ms   REAL,
    model        TEXT,
    llm_provider TEXT,
    max_tokens   INTEGER,
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass(frozen=True)
class QuerySpec:
    """Single query entry from queries.json."""

    idx: int
    text: str
    answer_type: str


@dataclass(frozen=True)
class DocRunResult:
    """Generation outcome for one document."""

    doc_id: str
    output_path: Path
    status: str
    query_idx: int = 0
    cache_hit: bool = False
    answer: Any = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    api_call_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class GenerationSummary:
    """Run summary returned by generate_ground_truth()."""

    dataset_root: Path
    query: QuerySpec
    selected_count: int
    generated_count: int
    skipped_existing_count: int
    failed_count: int
    run_latency_ms: float = 0.0
    log_path: Path | None = None
    results: tuple[DocRunResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BatchGenerationSummary:
    """Run summary returned by generate_ground_truth_for_queries()."""

    dataset_root: Path
    queries: tuple[QuerySpec, ...]
    selected_count: int
    generated_count: int
    skipped_existing_count: int
    failed_count: int
    run_latency_ms: float = 0.0
    log_path: Path | None = None
    results: tuple[DocRunResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AzureResponsesCallResult:
    """Raw Azure Responses API output plus provider-reported usage."""

    response: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class RunMetrics:
    """Deduplicated runtime metrics across generated result rows."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    local_cache_hits: int
    api_latency_ms: float


def generate_ground_truth(
    target_dir: str | Path,
    query_idx: int,
    num_doc: int | None = None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    *,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
    generation_mode: GenerationMode = DEFAULT_GENERATION_MODE,
    cache_db: str = DEFAULT_CACHE_DB_PATH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    progress_cost: bool = False,
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
) -> GenerationSummary:
    """Generate ground-truth answers for sampled PDFs.

    Args:
        target_dir: Dataset directory, either datasets/<name> or datasets/<name>/latest.
        query_idx: 1-based query index from queries.json.
        num_doc: Number of PDFs to sample before skipping existing GT; None means all.
        llm_provider: LLM provider.
        model: Model identifier, default gpt-5.4-mini.
        seed: Random sampling seed.
        input_mode: auto resolves to text; text uses extracted text;
            claude-read-pdf asks Claude Code to read the local PDF path.
        generation_mode: single answers this query with the single-answer schema;
            all answers the selected query set with the multi-answer schema.
        cache_db: SQLite LLM cache path.
        max_tokens: Max output tokens.
        temperature: Sampling temperature for Azure text-mode calls.
        claude_timeout_sec: Timeout for each Claude Code CLI call.
        progress_cost: Print per-call API usage/cost as the run progresses.
        log_dir: Optional directory for run log files.
    """
    run_t0 = time.perf_counter()
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    resolved_generation_mode = resolve_generation_mode(generation_mode)
    dataset_root = resolve_dataset_root(target_dir)
    query = load_query(dataset_root / "queries.json", query_idx)
    pdf_paths = sample_pdf_paths(dataset_root / "raw", num_doc=num_doc, seed=seed)
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    run_logger = _make_run_logger(
        log_dir=log_dir,
        dataset_root=dataset_root,
        llm_provider=resolved_provider,
        model=resolved_model,
        input_mode=resolved_input_mode,
        generation_mode=resolved_generation_mode,
    )

    text_caller = (
        AzureResponsesTextCacheCaller(cache_db)
        if resolved_provider == "azure" and resolved_input_mode == "text"
        else None
    )
    claude_caller = (
        ClaudeCodeCacheCaller(cache_db) if resolved_provider == "claude-code" else None
    )

    results: list[DocRunResult] = []
    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        output_path = gt_dir / f"{doc_id}.txt_answers.json"
        gt_key = str(query_idx)
        if output_path.exists() and _has_existing_answer(output_path, gt_key):
            _record_result(
                results,
                DocRunResult(
                    doc_id=doc_id,
                    output_path=output_path,
                    status="skipped_existing",
                    query_idx=query.idx,
                ),
                progress_cost=progress_cost,
            )
            continue

        response: CacheResult | None = None
        api_call_id = ""
        try:
            if resolved_generation_mode == "all":
                api_call_id = _build_result_api_call_id(
                    pdf_path=pdf_path,
                    queries=[query],
                    generation_mode=resolved_generation_mode,
                )
                response = _call_model_for_queries_for_doc(
                    pdf_path=pdf_path,
                    queries=[query],
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    claude_timeout_sec=claude_timeout_sec,
                )
                answer = parse_answers_response(response.response, [query])[query.idx]
            else:
                response = _call_model_for_doc(
                    pdf_path=pdf_path,
                    query=query,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    claude_timeout_sec=claude_timeout_sec,
                )
                answer = parse_answer_response(response.response)
                api_call_id = ""
            _merge_ground_truth(output_path, gt_key, answer)
            _record_result(
                results,
                _doc_run_result(
                    doc_id=doc_id,
                    output_path=output_path,
                    status="generated",
                    query_idx=query.idx,
                    response=response,
                    answer=answer,
                    api_call_id=api_call_id,
                ),
                progress_cost=progress_cost,
                run_logger=run_logger,
            )
        except Exception as exc:  # keep the batch moving across bad PDFs/API failures
            _record_result(
                results,
                _doc_run_result(
                    doc_id=doc_id,
                    output_path=output_path,
                    status="failed",
                    query_idx=query.idx,
                    response=response,
                    api_call_id=api_call_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                progress_cost=progress_cost,
                run_logger=run_logger,
            )
    summary = GenerationSummary(
        dataset_root=dataset_root,
        query=query,
        selected_count=len(pdf_paths),
        generated_count=sum(r.status == "generated" for r in results),
        skipped_existing_count=sum(r.status == "skipped_existing" for r in results),
        failed_count=sum(r.status == "failed" for r in results),
        run_latency_ms=_elapsed_ms(run_t0),
        log_path=run_logger.log_path if run_logger is not None else None,
        results=tuple(results),
    )
    if run_logger is not None:
        run_logger.log_run_summary(summary.results, summary.run_latency_ms)
        run_logger.close()
    return summary


def generate_ground_truth_for_queries(
    target_dir: str | Path,
    query_indices: Sequence[int] | None = None,
    num_doc: int | None = None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    *,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
    generation_mode: GenerationMode = DEFAULT_GENERATION_MODE,
    cache_db: str = DEFAULT_CACHE_DB_PATH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    progress_cost: bool = False,
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
) -> BatchGenerationSummary:
    """Generate ground-truth answers for multiple queries using doc-major order.

    This is the preferred entry point when running more than one query. It keeps
    the same document adjacent across query calls, which gives provider-side
    prompt caching the best chance to reuse the long document prefix.

    Args:
        progress_cost: Print per-call API usage/cost as the run progresses.
        log_dir: Optional directory for run log files.
    """
    run_t0 = time.perf_counter()
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    resolved_generation_mode = resolve_generation_mode(generation_mode)
    dataset_root = resolve_dataset_root(target_dir)
    queries = load_queries(dataset_root / "queries.json", query_indices)
    pdf_paths = sample_pdf_paths(dataset_root / "raw", num_doc=num_doc, seed=seed)
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    run_logger = _make_run_logger(
        log_dir=log_dir,
        dataset_root=dataset_root,
        llm_provider=resolved_provider,
        model=resolved_model,
        input_mode=resolved_input_mode,
        generation_mode=resolved_generation_mode,
    )

    text_caller = (
        AzureResponsesTextCacheCaller(cache_db)
        if resolved_provider == "azure" and resolved_input_mode == "text"
        else None
    )
    claude_caller = (
        ClaudeCodeCacheCaller(cache_db) if resolved_provider == "claude-code" else None
    )

    results: list[DocRunResult] = []
    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        output_path = gt_dir / f"{doc_id}.txt_answers.json"
        document_text = (
            _extract_pdf_text_for_prompt(pdf_path)
            if resolved_input_mode == "text"
            else None
        )

        if resolved_generation_mode == "all":
            pending_queries: list[QuerySpec] = []
            for query in queries:
                gt_key = str(query.idx)
                if output_path.exists() and _has_existing_answer(output_path, gt_key):
                    _record_result(
                        results,
                        DocRunResult(
                            doc_id=doc_id,
                            output_path=output_path,
                            status="skipped_existing",
                            query_idx=query.idx,
                        ),
                        progress_cost=progress_cost,
                    )
                else:
                    pending_queries.append(query)

            if not pending_queries:
                continue

            api_call_id = _build_result_api_call_id(
                pdf_path=pdf_path,
                queries=pending_queries,
                generation_mode=resolved_generation_mode,
            )
            response: CacheResult | None = None
            try:
                response = _call_model_for_queries_for_doc(
                    pdf_path=pdf_path,
                    queries=pending_queries,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    claude_timeout_sec=claude_timeout_sec,
                    document_text=document_text,
                )
                answers = parse_answers_response(response.response, pending_queries)
                _merge_ground_truth_answers(output_path, answers)
                for query in pending_queries:
                    _record_result(
                        results,
                        _doc_run_result(
                            doc_id=doc_id,
                            output_path=output_path,
                            status="generated",
                            query_idx=query.idx,
                            response=response,
                            answer=answers[query.idx],
                            api_call_id=api_call_id,
                        ),
                        progress_cost=progress_cost,
                        run_logger=run_logger,
                    )
            except Exception as exc:  # keep the batch moving across bad PDFs/API failures
                for query in pending_queries:
                    _record_result(
                        results,
                        _doc_run_result(
                            doc_id=doc_id,
                            output_path=output_path,
                            status="failed",
                            query_idx=query.idx,
                            response=response,
                            api_call_id=api_call_id,
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                        progress_cost=progress_cost,
                        run_logger=run_logger,
                    )
            continue

        for query in queries:
            gt_key = str(query.idx)
            if output_path.exists() and _has_existing_answer(output_path, gt_key):
                _record_result(
                    results,
                    DocRunResult(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="skipped_existing",
                        query_idx=query.idx,
                    ),
                    progress_cost=progress_cost,
                )
                continue

            response: CacheResult | None = None
            try:
                response = _call_model_for_doc(
                    pdf_path=pdf_path,
                    query=query,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    claude_timeout_sec=claude_timeout_sec,
                    document_text=document_text,
                )
                answer = parse_answer_response(response.response)
                _merge_ground_truth(output_path, gt_key, answer)
                _record_result(
                    results,
                    _doc_run_result(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="generated",
                        query_idx=query.idx,
                        response=response,
                        answer=answer,
                    ),
                    progress_cost=progress_cost,
                    run_logger=run_logger,
                )
            except Exception as exc:  # keep the batch moving across bad PDFs/API failures
                _record_result(
                    results,
                    _doc_run_result(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="failed",
                        query_idx=query.idx,
                        response=response,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    progress_cost=progress_cost,
                    run_logger=run_logger,
                )

    summary = BatchGenerationSummary(
        dataset_root=dataset_root,
        queries=tuple(queries),
        selected_count=len(pdf_paths),
        generated_count=sum(r.status == "generated" for r in results),
        skipped_existing_count=sum(r.status == "skipped_existing" for r in results),
        failed_count=sum(r.status == "failed" for r in results),
        run_latency_ms=_elapsed_ms(run_t0),
        log_path=run_logger.log_path if run_logger is not None else None,
        results=tuple(results),
    )
    if run_logger is not None:
        run_logger.log_run_summary(summary.results, summary.run_latency_ms)
        run_logger.close()
    return summary


class AzureResponsesTextCacheCaller:
    """Azure Responses API text caller with the shared SQLite LLM cache."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        conn = self._get_conn()
        conn.execute(_CACHE_CREATE_TABLE_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute(_CACHE_CREATE_TABLE_SQL)
        return self._local.conn

    def call(
        self,
        prompt: str,
        llm_provider: str,
        model: str,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> CacheResult:
        if llm_provider != "azure":
            raise ValueError("text Responses input mode supports only azure")
        resolved_model = _require_model(model)
        normalized_temperature = float(temperature)
        cache_key = _build_azure_text_cache_key(
            prompt=prompt,
            llm_provider=llm_provider,
            model=resolved_model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=normalized_temperature,
        )

        conn = self._get_conn()
        if normalized_temperature == 0.0:
            row = conn.execute(
                "SELECT response, input_tokens, output_tokens, latency_ms "
                "FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                response, input_tokens, output_tokens, latency_ms = row
                return CacheResult(
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cache_hit=True,
                )

        t0 = time.perf_counter()
        call_result = _azure_responses_text_call(
            prompt=prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=normalized_temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if normalized_temperature == 0.0:
            conn.execute(
                """
                INSERT OR IGNORE INTO llm_cache
                    (cache_key, prompt_text, response, input_tokens, output_tokens,
                     latency_ms, model, llm_provider, max_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    prompt,
                    call_result.response,
                    call_result.input_tokens,
                    call_result.output_tokens,
                    latency_ms,
                    resolved_model,
                    llm_provider,
                    max_tokens,
                ),
            )
            conn.commit()

        return CacheResult(
            response=call_result.response,
            input_tokens=call_result.input_tokens,
            output_tokens=call_result.output_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
            cached_input_tokens=call_result.cached_input_tokens,
            cost_usd=call_result.cost_usd,
        )


class ClaudeCodeCacheCaller:
    """Claude Code CLI caller with the shared SQLite LLM cache."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        conn = self._get_conn()
        conn.execute(_CACHE_CREATE_TABLE_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute(_CACHE_CREATE_TABLE_SQL)
        return self._local.conn

    def call(
        self,
        *,
        pdf_path: Path,
        prompt: str,
        llm_provider: str,
        model: str,
        input_mode: ResolvedInputMode,
        max_tokens: int,
        timeout_sec: int,
        response_schema: dict[str, Any] | None = None,
    ) -> CacheResult:
        if llm_provider != "claude-code":
            raise ValueError("Claude Code caller requires llm_provider='claude-code'")
        if input_mode not in {"text", "claude-read-pdf"}:
            raise ValueError(
                "Claude Code caller supports input_mode='text' or 'claude-read-pdf'"
            )
        resolved_model = _require_model(model)
        pdf_hash = _sha256_file(pdf_path)
        prompt_hash = _sha256_text(prompt)
        cache_key = _build_claude_code_cache_key(
            pdf_hash=pdf_hash,
            prompt_hash=prompt_hash,
            llm_provider=llm_provider,
            model=resolved_model,
            input_mode=input_mode,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            response_schema=response_schema,
        )

        conn = self._get_conn()
        row = conn.execute(
            "SELECT response, input_tokens, output_tokens, latency_ms "
            "FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is not None:
            response, input_tokens, output_tokens, latency_ms = row
            return CacheResult(
                response=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cache_hit=True,
            )

        t0 = time.perf_counter()
        response_text, input_tokens, output_tokens = _claude_code_call(
            pdf_path=pdf_path,
            prompt=prompt,
            model=resolved_model,
            input_mode=input_mode,
            timeout_sec=timeout_sec,
            response_schema=response_schema or _answer_response_schema(),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metadata = json.dumps(
            {
                "mode": "gt_gen_claude_code_v1",
                "input_mode": input_mode,
                "pdf_name": pdf_path.name,
                "pdf_sha256": pdf_hash,
                "prompt_sha256": prompt_hash,
                "response_schema": response_schema,
                "timeout_sec": timeout_sec,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO llm_cache
                (cache_key, prompt_text, response, input_tokens, output_tokens,
                 latency_ms, model, llm_provider, max_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                metadata,
                response_text,
                input_tokens,
                output_tokens,
                latency_ms,
                resolved_model,
                llm_provider,
                max_tokens,
            ),
        )
        conn.commit()

        return CacheResult(
            response=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
        )


def normalize_llm_provider(llm_provider: str) -> str:
    provider = str(llm_provider).strip().lower().replace("_", "-")
    aliases = {
        "azure": "azure",
        "claude": "claude-code",
        "claude-code": "claude-code",
        "claudecode": "claude-code",
    }
    if provider not in aliases:
        raise ValueError("llm_provider must be one of: azure, claude-code")
    return aliases[provider]


def resolve_model_for_provider(llm_provider: str, model: str) -> str:
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = _require_model(model)
    if resolved_provider == "claude-code" and resolved_model == DEFAULT_MODEL:
        return DEFAULT_CLAUDE_MODEL
    return resolved_model


def resolve_input_mode(llm_provider: str, input_mode: InputMode) -> ResolvedInputMode:
    resolved_provider = normalize_llm_provider(llm_provider)
    mode = str(input_mode).strip().lower().replace("_", "-")
    if mode == "auto":
        return "text"
    if mode == "text":
        return "text"
    if mode == "claude-read-pdf":
        if resolved_provider != "claude-code":
            raise ValueError(
                "input_mode='claude-read-pdf' is supported only for claude-code"
            )
        return "claude-read-pdf"
    raise ValueError("input_mode must be one of: auto, text, claude-read-pdf")


def resolve_generation_mode(generation_mode: str) -> GenerationMode:
    mode = str(generation_mode).strip().lower().replace("_", "-")
    if mode in {"single", "all"}:
        return mode  # type: ignore[return-value]
    raise ValueError("generation_mode must be one of: single, all")


def resolve_dataset_root(target_dir: str | Path) -> Path:
    """Resolve datasets/<name> or datasets/<name>/latest to the latest root."""
    path = Path(target_dir).expanduser().resolve()
    candidates = [path, path / "latest"]
    for candidate in candidates:
        if (candidate / "raw").is_dir() and (candidate / "queries.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"{target_dir} must contain raw/ and queries.json, or contain latest/raw and latest/queries.json"
    )


def load_query(queries_path: Path, query_idx: int) -> QuerySpec:
    if query_idx < 1:
        raise ValueError("query_idx is 1-based and must be >= 1")
    with queries_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{queries_path} must contain a JSON list")
    zero_based_idx = query_idx - 1
    if zero_based_idx >= len(data):
        raise ValueError(
            f"query_idx={query_idx} out of range for {queries_path} ({len(data)} queries)"
        )
    entry = data[zero_based_idx]
    if not isinstance(entry, dict):
        raise ValueError(f"query {query_idx} must be a JSON object")
    text = str(entry.get("text", "")).strip()
    answer_type = str(entry.get("answer_type", "")).strip() or "string"
    if not text:
        raise ValueError(f"query {query_idx} has empty text")
    return QuerySpec(idx=query_idx, text=text, answer_type=answer_type)


def load_queries(
    queries_path: Path, query_indices: Sequence[int] | None = None
) -> list[QuerySpec]:
    with queries_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{queries_path} must contain a JSON list")
    if query_indices is None:
        selected = range(1, len(data) + 1)
    else:
        selected = query_indices
    queries: list[QuerySpec] = []
    for query_idx in selected:
        if query_idx < 1:
            raise ValueError("query indices are 1-based and must be >= 1")
        zero_based_idx = query_idx - 1
        if zero_based_idx >= len(data):
            raise ValueError(
                f"query_idx={query_idx} out of range for {queries_path} ({len(data)} queries)"
            )
        entry = data[zero_based_idx]
        if not isinstance(entry, dict):
            raise ValueError(f"query {query_idx} must be a JSON object")
        text = str(entry.get("text", "")).strip()
        answer_type = str(entry.get("answer_type", "")).strip() or "string"
        if not text:
            raise ValueError(f"query {query_idx} has empty text")
        queries.append(QuerySpec(idx=query_idx, text=text, answer_type=answer_type))
    return queries


def sample_pdf_paths(raw_dir: Path, num_doc: int | None, seed: int) -> list[Path]:
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {raw_dir}")
    if num_doc is None:
        return pdf_paths
    if num_doc < 0:
        raise ValueError("num_doc must be non-negative")
    if num_doc == 0:
        return []
    if num_doc > len(pdf_paths):
        raise ValueError(f"num_doc={num_doc} exceeds available PDFs ({len(pdf_paths)})")
    return sorted(random.Random(seed).sample(pdf_paths, num_doc))


def parse_answer_response(raw_response: str) -> Any:
    data = _loads_json_object(raw_response)
    if "answer" not in data:
        raise ValueError("LLM response JSON must contain an 'answer' field")
    return _normalize_answer_value(data["answer"])


def parse_answers_response(
    raw_response: str, queries: Sequence[QuerySpec]
) -> dict[int, Any]:
    data = _loads_json_object(raw_response)
    answers = data.get("answers")
    if not isinstance(answers, list):
        raise ValueError("LLM response JSON must contain an 'answers' list")

    expected = {query.idx for query in queries}
    parsed: dict[int, Any] = {}
    for entry in answers:
        if not isinstance(entry, dict):
            raise ValueError("Each answers entry must be a JSON object")
        raw_query_idx = entry.get("query_idx")
        if isinstance(raw_query_idx, int):
            query_idx = raw_query_idx
        elif isinstance(raw_query_idx, str) and raw_query_idx.isdigit():
            query_idx = int(raw_query_idx)
        else:
            raise ValueError("Each answers entry must include an integer query_idx")
        if query_idx not in expected:
            raise ValueError(f"Unexpected answer for query_idx={query_idx}")
        if query_idx in parsed:
            raise ValueError(f"Duplicate answer for query_idx={query_idx}")
        if "answer" not in entry:
            raise ValueError(f"Answer entry for query_idx={query_idx} is missing answer")
        parsed[query_idx] = _normalize_answer_value(entry["answer"])

    missing = sorted(expected - set(parsed))
    if missing:
        raise ValueError(f"Missing answers for query indices: {missing}")
    return parsed


def _normalize_answer_value(answer: Any) -> Any:
    if answer is None:
        return "None"
    if isinstance(answer, str):
        stripped = answer.strip()
        return stripped if stripped else "None"
    return answer


def _call_model_for_doc(
    *,
    pdf_path: Path,
    query: QuerySpec,
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    temperature: float = 0.0,
    document_text: str | None = None,
) -> CacheResult:
    dataset_name = _dataset_name_for_log(_dataset_root_from_pdf_path(pdf_path))
    prompt_template = _load_gt_prompt_template(dataset_name)
    examples = _load_gt_examples(dataset_name)
    prompt = build_ground_truth_prompt(
        query=query,
        doc_id=pdf_path.stem,
        prompt_template=prompt_template,
        examples=examples,
    )
    response_schema = _answer_response_schema()
    if input_mode == "text":
        resolved_document_text = (
            document_text
            if document_text is not None
            else _extract_pdf_text_for_prompt(pdf_path)
        )
        text_prompt = build_ground_truth_text_prompt(
            query=query,
            doc_id=pdf_path.stem,
            document_text=resolved_document_text,
            prompt_template=prompt_template,
            examples=examples,
        )
        if llm_provider == "claude-code":
            if claude_caller is None:
                raise RuntimeError("Claude Code caller is not initialized")
            return claude_caller.call(
                pdf_path=pdf_path,
                prompt=text_prompt,
                llm_provider=llm_provider,
                model=model,
                input_mode=input_mode,
                max_tokens=max_tokens,
                timeout_sec=claude_timeout_sec,
                response_schema=response_schema,
            )
        if text_caller is None:
            raise RuntimeError("text caller is not initialized")
        return text_caller.call(
            text_prompt,
            llm_provider=llm_provider,
            model=model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=temperature,
        )
    if input_mode == "claude-read-pdf":
        if claude_caller is None:
            raise RuntimeError("Claude Code caller is not initialized")
        prompt_with_path = (
            prompt
            + "\n\n"
            + f"Local PDF path: {pdf_path}\n"
            + "Use the Read tool to inspect this local PDF file. Do not use outside knowledge."
        )
        return claude_caller.call(
            pdf_path=pdf_path,
            prompt=prompt_with_path,
            llm_provider=llm_provider,
            model=model,
            input_mode=input_mode,
            max_tokens=max_tokens,
            timeout_sec=claude_timeout_sec,
            response_schema=response_schema,
        )
    raise ValueError(f"Unsupported input_mode={input_mode!r}")


def _call_model_for_queries_for_doc(
    *,
    pdf_path: Path,
    queries: Sequence[QuerySpec],
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    temperature: float = 0.0,
    document_text: str | None = None,
) -> CacheResult:
    dataset_name = _dataset_name_for_log(_dataset_root_from_pdf_path(pdf_path))
    prompt_template = _load_gt_prompt_template(dataset_name)
    examples = _load_gt_examples(dataset_name)
    prompt = build_ground_truth_all_prompt(
        queries=queries,
        doc_id=pdf_path.stem,
        prompt_template=prompt_template,
        examples=examples,
    )
    response_schema = _answers_response_schema(queries)
    if input_mode == "text":
        resolved_document_text = (
            document_text
            if document_text is not None
            else _extract_pdf_text_for_prompt(pdf_path)
        )
        text_prompt = build_ground_truth_all_text_prompt(
            queries=queries,
            doc_id=pdf_path.stem,
            document_text=resolved_document_text,
            prompt_template=prompt_template,
            examples=examples,
        )
        if llm_provider == "claude-code":
            if claude_caller is None:
                raise RuntimeError("Claude Code caller is not initialized")
            return claude_caller.call(
                pdf_path=pdf_path,
                prompt=text_prompt,
                llm_provider=llm_provider,
                model=model,
                input_mode=input_mode,
                max_tokens=max_tokens,
                timeout_sec=claude_timeout_sec,
                response_schema=response_schema,
            )
        if text_caller is None:
            raise RuntimeError("text caller is not initialized")
        return text_caller.call(
            text_prompt,
            llm_provider=llm_provider,
            model=model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=temperature,
        )
    if input_mode == "claude-read-pdf":
        if claude_caller is None:
            raise RuntimeError("Claude Code caller is not initialized")
        prompt_with_path = (
            prompt
            + "\n\n"
            + f"Local PDF path: {pdf_path}\n"
            + "Use the Read tool to inspect this local PDF file. Do not use outside knowledge."
        )
        return claude_caller.call(
            pdf_path=pdf_path,
            prompt=prompt_with_path,
            llm_provider=llm_provider,
            model=model,
            input_mode=input_mode,
            max_tokens=max_tokens,
            timeout_sec=claude_timeout_sec,
            response_schema=response_schema,
        )
    raise ValueError(f"Unsupported input_mode={input_mode!r}")


def build_ground_truth_prompt(
    *,
    query: QuerySpec,
    doc_id: str,
    prompt_template: str | None = None,
    examples: dict[str, Any] | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="SINGLE",
        doc_id=doc_id,
        query_idx=query.idx,
        query_text=query.text,
        answer_type=query.answer_type,
        one_shot_example_block=_format_single_example_block(
            examples, query_idx=query.idx, live_doc_id=doc_id
        ),
    )


def build_ground_truth_text_prompt(
    *,
    query: QuerySpec,
    doc_id: str,
    document_text: str,
    prompt_template: str | None = None,
    examples: dict[str, Any] | None = None,
) -> str:
    """Build a cache-friendly text prompt with document before query details."""
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="SINGLE_TEXT",
        doc_id=doc_id,
        document_text=document_text,
        query_idx=query.idx,
        query_text=query.text,
        answer_type=query.answer_type,
        one_shot_example_block=_format_single_example_block(
            examples, query_idx=query.idx, live_doc_id=doc_id
        ),
    )


def build_ground_truth_all_prompt(
    *,
    queries: Sequence[QuerySpec],
    doc_id: str,
    prompt_template: str | None = None,
    examples: dict[str, Any] | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="ALL",
        doc_id=doc_id,
        queries=_format_query_list(queries),
        one_shot_example_block=_format_all_example_block(
            examples, live_doc_id=doc_id
        ),
    )


def build_ground_truth_all_text_prompt(
    *,
    queries: Sequence[QuerySpec],
    doc_id: str,
    document_text: str,
    prompt_template: str | None = None,
    examples: dict[str, Any] | None = None,
) -> str:
    """Build a cache-friendly multi-query text prompt with document before queries."""
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="ALL_TEXT",
        doc_id=doc_id,
        document_text=document_text,
        queries=_format_query_list(queries),
        one_shot_example_block=_format_all_example_block(
            examples, live_doc_id=doc_id
        ),
    )


def _render_gt_prompt(
    *,
    prompt_template: str | None,
    section: str,
    **values: Any,
) -> str:
    template = prompt_template or _load_gt_prompt_template(None)
    prompt = _gt_prompt_section(template, section)
    for key, value in values.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt.strip() + "\n"


def _load_gt_prompt_template(dataset_name: str | None) -> str:
    prompt_path = _gt_prompt_path(dataset_name)
    return prompt_path.read_text(encoding="utf-8")


def _gt_prompt_path(dataset_name: str | None) -> Path:
    if dataset_name:
        candidate = GT_PROMPT_DIR / f"{_safe_slug(dataset_name).lower()}.txt"
        if candidate.is_file():
            return candidate
    default_path = GT_PROMPT_DIR / DEFAULT_GT_PROMPT_NAME
    if not default_path.is_file():
        raise FileNotFoundError(f"Missing default GT prompt template: {default_path}")
    return default_path


def _gt_prompt_section(prompt_template: str, section: str) -> str:
    section_name = section.strip().upper()
    start_marker = f"[{section_name}]"
    end_marker = f"[/{section_name}]"
    start = prompt_template.find(start_marker)
    if start < 0:
        raise ValueError(f"GT prompt template is missing section {start_marker}")
    start += len(start_marker)
    end = prompt_template.find(end_marker, start)
    if end < 0:
        raise ValueError(f"GT prompt template is missing section {end_marker}")
    prompt = prompt_template[start:end].strip()
    if not prompt:
        raise ValueError(f"GT prompt template section {start_marker} is empty")
    return prompt


def _load_gt_examples(dataset_name: str | None) -> dict[str, Any] | None:
    """Load optional per-query gold examples for ``dataset_name``."""
    if not dataset_name:
        return None
    candidate = GT_PROMPT_DIR / f"{_safe_slug(dataset_name).lower()}_examples.json"
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _example_block_applies(
    examples: dict[str, Any] | None, live_doc_id: str
) -> bool:
    if not examples or not isinstance(examples.get("examples"), dict):
        return False
    # Guard against feeding the model the literal answer key for the same doc.
    if examples.get("doc_id") == live_doc_id:
        return False
    return True


def _format_single_example_block(
    examples: dict[str, Any] | None,
    *,
    query_idx: int,
    live_doc_id: str,
) -> str:
    if not _example_block_applies(examples, live_doc_id):
        return ""
    entry = examples["examples"].get(str(query_idx))
    if not entry:
        return ""
    response_json = json.dumps(
        {
            "reasoning": entry["reasoning"],
            "answer": entry["answer"],
        },
        ensure_ascii=False,
    )
    return (
        "One-shot example (from a reference NOPV document; apply analogous"
        " reasoning to the current document, do not copy verbatim):\n"
        f"Document id: {examples['doc_id']}\n"
        f"Question index: {query_idx}\n"
        f"Question: {entry['query']}\n"
        f"Answer type: {entry['answer_type']}\n"
        "Response:\n"
        f"{response_json}\n"
        "End of example."
    )


def _format_all_example_block(
    examples: dict[str, Any] | None,
    *,
    live_doc_id: str,
) -> str:
    if not _example_block_applies(examples, live_doc_id):
        return ""
    all_ids = examples.get("all_example_query_ids") or []
    selected_specs: list[QuerySpec] = []
    selected_entries: list[tuple[int, dict[str, Any]]] = []
    for idx in all_ids:
        entry = examples["examples"].get(str(idx))
        if not entry:
            continue
        selected_specs.append(
            QuerySpec(
                idx=int(idx),
                text=entry["query"],
                answer_type=entry["answer_type"],
            )
        )
        selected_entries.append((int(idx), entry))
    if not selected_entries:
        return ""
    question_block = _format_query_list(selected_specs).rstrip()
    response_json = json.dumps(
        {
            "answers": [
                {
                    "query_idx": idx,
                    "reasoning": entry["reasoning"],
                    "answer": entry["answer"],
                }
                for idx, entry in selected_entries
            ]
        },
        ensure_ascii=False,
    )
    return (
        "One-shot example (from a reference NOPV document; apply analogous"
        " reasoning, do not copy verbatim):\n"
        f"Document id: {examples['doc_id']}\n"
        "[QUESTIONS]\n"
        f"{question_block}\n"
        "Response:\n"
        f"{response_json}\n"
        "End of example."
    )


def _dataset_root_from_pdf_path(pdf_path: Path) -> Path:
    return pdf_path.parent.parent


def _format_query_list(queries: Sequence[QuerySpec]) -> str:
    return "\n".join(
        (
            f"Question index: {query.idx}\n"
            f"Question: {query.text}\n"
            f"Answer type: {query.answer_type}\n"
        )
        for query in queries
    )


def _extract_pdf_text_for_prompt(pdf_path: Path) -> str:
    import fitz

    parts = ["[DOCUMENT TEXT START]"]
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                parts.append(f"[Page {page_index}]\n{text}")
    parts.append("[DOCUMENT TEXT END]")
    return "\n\n".join(parts)


def _azure_responses_text_call(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    response_schema: dict[str, Any] | None,
    temperature: float,
) -> AzureResponsesCallResult:
    client, deployment = _azure_responses_client_and_deployment(model)
    response = client.responses.create(
        model=deployment,
        input=prompt,
        max_output_tokens=max_tokens,
        text=_responses_text_config(response_schema, name="gt_gen_single_answer"),
        temperature=temperature,
    )
    answer = str(getattr(response, "output_text", "")).strip()
    if not answer:
        raise ValueError("Azure Responses API returned empty output_text")
    usage = getattr(response, "usage", None)
    return _build_azure_call_result(
        answer=answer,
        usage=usage,
        model=model,
    )


def _azure_responses_client_and_deployment(model: str) -> tuple[OpenAI, str]:
    env_prefix = _azure_env_prefix_for_model(model)
    api_key = _require_env(f"{env_prefix}_API_KEY")
    endpoint = _require_env(f"{env_prefix}_API_BASE").rstrip("/")
    deployment = _require_env(f"{env_prefix}_DEPLOYMENT")
    client = OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=api_key,
    )
    return client, deployment


def _build_azure_call_result(
    *,
    answer: str,
    usage: Any,
    model: str,
) -> AzureResponsesCallResult:
    if usage is None:
        raise ValueError("Azure Responses API did not return usage")
    input_tokens = _usage_int(usage, "input_tokens", fallback=-1)
    if input_tokens < 0:
        raise ValueError("Azure Responses API usage did not include input_tokens")
    cached_input_tokens = min(
        _usage_cached_input_tokens(usage),
        input_tokens,
    )
    output_tokens = _usage_int(
        usage,
        "output_tokens",
        fallback=-1,
    )
    if output_tokens < 0:
        raise ValueError("Azure Responses API usage did not include output_tokens")
    cost_usd = compute_cost_with_cached_input(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        llm_provider="azure",
        model=model,
    )
    return AzureResponsesCallResult(
        response=answer,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _claude_code_call(
    *,
    pdf_path: Path,
    prompt: str,
    model: str,
    input_mode: ResolvedInputMode,
    timeout_sec: int,
    response_schema: dict[str, Any],
) -> tuple[str, int, int]:
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--json-schema",
        json.dumps(response_schema, separators=(",", ":")),
    ]
    if input_mode == "claude-read-pdf":
        cmd.extend(["--tools=Read", "--add-dir", str(pdf_path.parent)])

    completed = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    wrapper = _parse_claude_cli_stdout(completed.stdout)
    if completed.returncode != 0 or wrapper.get("is_error"):
        errors = wrapper.get("errors") or wrapper.get("result") or completed.stderr
        raise RuntimeError(f"Claude Code failed: {errors}")

    result = wrapper.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Claude Code returned empty result")

    usage = wrapper.get("usage")
    input_tokens = _claude_usage_token_total(usage, fallback=estimate_tokens(prompt))
    output_tokens = _claude_usage_token(
        usage,
        "output_tokens",
        fallback=estimate_tokens(result),
    )
    return result.strip(), input_tokens, output_tokens


def _answer_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "answer": _answer_value_schema(),
        },
        "required": ["reasoning", "answer"],
    }


def _answers_response_schema(queries: Sequence[QuerySpec]) -> dict[str, Any]:
    allowed_query_indices = [query.idx for query in queries]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answers": {
                "type": "array",
                "minItems": len(allowed_query_indices),
                "maxItems": len(allowed_query_indices),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query_idx": {"type": "integer", "enum": allowed_query_indices},
                        "reasoning": {"type": "string"},
                        "answer": _answer_value_schema(),
                    },
                    "required": ["query_idx", "reasoning", "answer"],
                },
            }
        },
        "required": ["answers"],
    }


def _answer_value_schema() -> dict[str, Any]:
    scalar_types = ["string", "integer", "number", "boolean", "null"]
    return {
        "type": scalar_types + ["array"],
        "items": {"type": scalar_types},
    }


def _responses_text_config(
    response_schema: dict[str, Any] | None,
    *,
    name: str,
) -> dict[str, Any] | None:
    if response_schema is None:
        return None
    schema_name = (
        "gt_gen_multi_answers"
        if "answers" in response_schema.get("properties", {})
        else name
    )
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "schema": response_schema,
        }
    }


def _parse_claude_cli_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Claude Code stdout did not contain JSON: {text[:500]}")
    decoder = json.JSONDecoder()
    data, _end = decoder.raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError("Claude Code stdout JSON must be an object")
    return data


def _claude_usage_token_total(usage: Any, fallback: int) -> int:
    if not isinstance(usage, dict):
        return fallback
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    total = sum(value for key in keys if isinstance((value := usage.get(key)), int))
    return total if total > 0 else fallback


def _claude_usage_token(usage: Any, key: str, fallback: int) -> int:
    if isinstance(usage, dict) and isinstance(usage.get(key), int):
        return usage[key]
    return fallback


def _usage_int(usage: Any, attr: str, fallback: int) -> int:
    if usage is None:
        return fallback
    value = getattr(usage, attr, None)
    if isinstance(value, int):
        return value
    if isinstance(usage, dict) and isinstance(usage.get(attr), int):
        return usage[attr]
    return fallback


def _usage_cached_input_tokens(usage: Any) -> int:
    details = getattr(usage, "input_tokens_details", None)
    value = _usage_int(details, "cached_tokens", fallback=0)
    if value:
        return value
    if isinstance(usage, dict):
        dict_details = usage.get("input_tokens_details")
        return _usage_int(dict_details, "cached_tokens", fallback=0)
    return 0


def _azure_env_prefix_for_model(model: str) -> str:
    resolved_model = _require_model(model)
    if resolved_model.startswith("gpt-5.4-mini"):
        return "AZURE_54MINI"
    if resolved_model.startswith("gpt-5.4"):
        return "AZURE_54"
    raise ValueError(
        f"Unsupported azure model={resolved_model!r}; supported: gpt-5.4-mini, gpt-5.4"
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _require_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be specified explicitly")
    resolved_model = model.strip()
    if resolved_model == "unspec" + "ified":
        raise ValueError("model uses a reserved invalid name")
    return resolved_model


def _build_azure_text_cache_key(
    *,
    prompt: str,
    llm_provider: str,
    model: str,
    max_tokens: int,
    response_schema: dict[str, Any] | None,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "mode": "gt_gen_azure_responses_text_v1",
            "prompt": prompt,
            "llm_provider": llm_provider,
            "model": model,
            "max_tokens": max_tokens,
            "response_schema": _schema_identity(response_schema),
            "temperature": temperature,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_claude_code_cache_key(
    *,
    pdf_hash: str,
    prompt_hash: str,
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    max_tokens: int,
    timeout_sec: int,
    response_schema: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        {
            "mode": "gt_gen_claude_code_v1",
            "pdf_sha256": pdf_hash,
            "prompt_sha256": prompt_hash,
            "llm_provider": llm_provider,
            "model": model,
            "input_mode": input_mode,
            "max_tokens": max_tokens,
            "response_schema": _schema_identity(response_schema),
            "timeout_sec": timeout_sec,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_identity(response_schema: dict[str, Any] | None) -> str:
    if response_schema is None:
        return ""
    return json.dumps(
        response_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _loads_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


def _has_existing_answer(output_path: Path, gt_key: str) -> bool:
    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or gt_key not in data:
        return False
    return _is_non_empty_answer(data[gt_key])


def _is_non_empty_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _merge_ground_truth(output_path: Path, gt_key: str, answer: Any) -> None:
    data: dict[str, Any] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{output_path} must contain a JSON object")
        data = loaded
    data[gt_key] = answer
    _atomic_write_json(output_path, data)


def _merge_ground_truth_answers(output_path: Path, answers: dict[int, Any]) -> None:
    data: dict[str, Any] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{output_path} must contain a JSON object")
        data = loaded
    for query_idx, answer in answers.items():
        data[str(query_idx)] = answer
    _atomic_write_json(output_path, data)


def _atomic_write_json(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_path = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PDF ground-truth answers")
    parser.add_argument(
        "--llm-provider",
        default=DEFAULT_LLM_PROVIDER,
        choices=["azure", "claude-code", "claude"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-dir", required=True)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-idx", type=int)
    query_group.add_argument(
        "--query-indices",
        help="Comma-separated 1-based query indices, ranges like 1-5, or 'all'.",
    )
    parser.add_argument("--num-doc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-mode",
        choices=["auto", "text", "claude-read-pdf"],
        default=DEFAULT_INPUT_MODE,
    )
    parser.add_argument(
        "--generation-mode",
        choices=["single", "all"],
        default=DEFAULT_GENERATION_MODE,
        help=(
            "single answers one question per API call; all answers selected "
            "questions per document in one API call."
        ),
    )
    parser.add_argument("--cache-db", default=DEFAULT_CACHE_DB_PATH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Azure text-mode calls; default 0.0.",
    )
    parser.add_argument(
        "--claude-timeout-sec",
        type=int,
        default=DEFAULT_CLAUDE_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--progress-cost",
        action="store_true",
        help="Print per-call API token usage and cumulative cost while running.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for gt_gen latency/cost log files.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> GenerationSummary | BatchGenerationSummary:
    args = _build_parser().parse_args(argv)
    if args.query_indices or args.generation_mode == "all":
        query_indices = (
            _parse_query_indices_arg(args.query_indices)
            if args.query_indices
            else [args.query_idx]
        )
        summary = generate_ground_truth_for_queries(
            target_dir=args.target_dir,
            query_indices=query_indices,
            num_doc=args.num_doc,
            llm_provider=args.llm_provider,
            model=args.model,
            seed=args.seed,
            input_mode=args.input_mode,
            generation_mode=args.generation_mode,
            cache_db=args.cache_db,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            claude_timeout_sec=args.claude_timeout_sec,
            progress_cost=args.progress_cost,
            log_dir=args.log_dir,
        )
        _print_batch_summary(summary)
        return summary

    summary = generate_ground_truth(
        target_dir=args.target_dir,
        query_idx=args.query_idx,
        num_doc=args.num_doc,
        llm_provider=args.llm_provider,
        model=args.model,
        seed=args.seed,
        input_mode=args.input_mode,
        generation_mode=args.generation_mode,
        cache_db=args.cache_db,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        claude_timeout_sec=args.claude_timeout_sec,
        progress_cost=args.progress_cost,
        log_dir=args.log_dir,
    )
    _print_summary(summary)
    return summary


def _parse_query_indices_arg(raw: str) -> list[int] | None:
    value = raw.strip().lower()
    if value == "all":
        return None
    indices: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"Invalid descending query range: {token}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    if not indices:
        raise ValueError("--query-indices must not be empty")
    return indices


class GTGenRunLogger:
    """File logger for one gt_gen run."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        dataset_root: Path,
        llm_provider: str,
        model: str,
        input_mode: ResolvedInputMode,
        generation_mode: str,
        tag: str = "gt_gen",
    ) -> None:
        log_path = _build_run_log_path(
            log_dir=log_dir,
            dataset_root=dataset_root,
            generation_mode=generation_mode,
            tag=tag,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.dataset_root = dataset_root
        self.llm_provider = llm_provider
        self.model = model
        self.input_mode = input_mode
        self.generation_mode = generation_mode
        self._seen_call_ids: set[str] = set()
        self._logger = logging.getLogger(f"gt_gen.run.{time.time_ns()}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler = logging.FileHandler(log_path, encoding="utf-8")
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self._logger.addHandler(self._handler)
        self._logger.info(
            _format_log_fields(
                event="run_start",
                dataset_root=str(dataset_root),
                llm_provider=llm_provider,
                model=model,
                input_mode=input_mode,
                generation_mode=generation_mode,
            )
        )

    def log_result(self, result: DocRunResult) -> None:
        if result.status == "skipped_existing":
            return
        call_id = _result_call_id(result)
        if call_id in self._seen_call_ids:
            return
        self._seen_call_ids.add(call_id)
        event = "llm_call"
        if result.cache_hit:
            event = "local_cache_hit"
        elif result.status == "failed" and not _result_has_call_metrics(result):
            event = "llm_call_failed"
        self._logger.info(
            _format_log_fields(
                event=event,
                dataset_root=str(self.dataset_root),
                doc_id=result.doc_id,
                query_ids=_result_query_ids(result),
                llm_provider=self.llm_provider,
                model=self.model,
                input_mode=self.input_mode,
                generation_mode=self.generation_mode,
                cache_hit=str(result.cache_hit).lower(),
                latency_ms=f"{result.latency_ms:.3f}",
                cost_usd=f"{result.cost_usd:.6f}",
                input_tokens=result.input_tokens,
                cached_input_tokens=result.cached_input_tokens,
                output_tokens=result.output_tokens,
                status=result.status,
                error=result.error,
            )
        )

    def log_run_summary(
        self,
        results: Sequence[DocRunResult],
        run_latency_ms: float,
    ) -> None:
        metrics = _run_metrics(results)
        self._logger.info(
            _format_log_fields(
                event="run_summary",
                dataset_root=str(self.dataset_root),
                llm_provider=self.llm_provider,
                model=self.model,
                input_mode=self.input_mode,
                generation_mode=self.generation_mode,
                run_latency_ms=f"{run_latency_ms:.3f}",
                api_latency_ms=f"{metrics.api_latency_ms:.3f}",
                cost_usd=f"{metrics.cost_usd:.6f}",
                input_tokens=metrics.input_tokens,
                cached_input_tokens=metrics.cached_input_tokens,
                output_tokens=metrics.output_tokens,
                local_cache_hits=metrics.local_cache_hits,
            )
        )

    def close(self) -> None:
        self._handler.flush()
        self._logger.removeHandler(self._handler)
        self._handler.close()


def _make_run_logger(
    *,
    log_dir: str | Path | None,
    dataset_root: Path,
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    generation_mode: str,
    tag: str = "gt_gen",
) -> GTGenRunLogger | None:
    if log_dir is None:
        return None
    return GTGenRunLogger(
        log_dir=log_dir,
        dataset_root=dataset_root,
        llm_provider=llm_provider,
        model=model,
        input_mode=input_mode,
        generation_mode=generation_mode,
        tag=tag,
    )


def _build_run_log_path(
    *,
    log_dir: str | Path,
    dataset_root: Path,
    generation_mode: str,
    tag: str = "gt_gen",
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dataset_slug = _safe_slug(_dataset_name_for_log(dataset_root))
    return Path(log_dir) / f"{tag}_{timestamp}_{dataset_slug}_{generation_mode}.log"


def _dataset_name_for_log(dataset_root: Path) -> str:
    return (
        dataset_root.parent.name
        if dataset_root.name == "latest"
        else dataset_root.name
    )


def _safe_slug(value: str) -> str:
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(chars).strip("_") or "dataset"


def _format_log_fields(**fields: Any) -> str:
    return " ".join(
        f"{key}={_format_log_value(value)}" for key, value in fields.items()
    )


def _format_log_value(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or '"' in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _result_call_id(result: DocRunResult) -> str:
    if result.api_call_id:
        return result.api_call_id
    return f"single:{result.doc_id}:{result.query_idx}:{result.output_path}"


def _result_query_ids(result: DocRunResult) -> str:
    if result.api_call_id:
        return result.api_call_id.rsplit(":", 1)[-1]
    return str(result.query_idx)


def _record_result(
    results: list[DocRunResult],
    result: DocRunResult,
    *,
    progress_cost: bool,
    run_logger: GTGenRunLogger | None = None,
) -> None:
    results.append(result)
    if run_logger is not None:
        run_logger.log_result(result)
    if progress_cost:
        _print_progress_cost(result, results)


def _doc_run_result(
    *,
    doc_id: str,
    output_path: Path,
    status: str,
    query_idx: int,
    response: CacheResult | None = None,
    answer: Any = None,
    api_call_id: str = "",
    error: str = "",
) -> DocRunResult:
    return DocRunResult(
        doc_id=doc_id,
        output_path=output_path,
        status=status,
        query_idx=query_idx,
        cache_hit=response.cache_hit if response is not None else False,
        answer=answer,
        input_tokens=response.input_tokens if response is not None else 0,
        cached_input_tokens=(
            response.cached_input_tokens if response is not None else 0
        ),
        output_tokens=response.output_tokens if response is not None else 0,
        cost_usd=response.cost_usd if response is not None else 0.0,
        latency_ms=response.latency_ms if response is not None else 0.0,
        api_call_id=api_call_id,
        error=error,
    )


def _build_result_api_call_id(
    *,
    pdf_path: Path,
    queries: Sequence[QuerySpec],
    generation_mode: GenerationMode,
) -> str:
    query_part = ",".join(str(query.idx) for query in queries)
    return f"{generation_mode}:{pdf_path.stem}:{query_part}"


def _api_usage_totals(
    results: Sequence[DocRunResult],
) -> tuple[int, int, int, float, int]:
    metrics = _run_metrics(results)
    return (
        metrics.input_tokens,
        metrics.cached_input_tokens,
        metrics.output_tokens,
        metrics.cost_usd,
        metrics.local_cache_hits,
    )


def _run_metrics(results: Sequence[DocRunResult]) -> RunMetrics:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    local_cache_hits = 0
    api_latency_ms = 0.0
    seen_usage_keys: set[str] = set()
    for position, result in enumerate(results):
        if result.status not in {"generated", "failed"}:
            continue
        usage_key = result.api_call_id or f"result:{position}"
        if usage_key in seen_usage_keys:
            continue
        seen_usage_keys.add(usage_key)
        if result.cache_hit:
            local_cache_hits += 1
            continue
        input_tokens += result.input_tokens
        cached_input_tokens += result.cached_input_tokens
        output_tokens += result.output_tokens
        cost_usd += result.cost_usd
        api_latency_ms += result.latency_ms
    return RunMetrics(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        local_cache_hits=local_cache_hits,
        api_latency_ms=api_latency_ms,
    )


def _result_has_call_metrics(result: DocRunResult) -> bool:
    return any(
        (
            result.input_tokens,
            result.cached_input_tokens,
            result.output_tokens,
            result.cost_usd,
            result.latency_ms,
        )
    )


def _format_usd(value: float) -> str:
    return f"${value:.6f}"


def _print_progress_cost(
    result: DocRunResult,
    results: Sequence[DocRunResult],
) -> None:
    if result.api_call_id and any(
        previous.api_call_id == result.api_call_id for previous in results[:-1]
    ):
        return
    input_tokens, cached_input_tokens, output_tokens, cost_usd, _local_hits = (
        _api_usage_totals(results)
    )
    total = _format_usd(cost_usd)
    if result.status == "failed":
        if _result_has_call_metrics(result):
            print(
                f"[GT_COST] q{result.query_idx} {result.doc_id}: "
                f"failed latency_ms={result.latency_ms:.3f} "
                f"cost={_format_usd(result.cost_usd)} total={total}"
            )
        else:
            print(f"[GT_COST] q{result.query_idx} {result.doc_id}: failed total={total}")
        return
    if result.status == "skipped_existing":
        print(f"[GT_COST] q{result.query_idx} {result.doc_id}: skipped total={total}")
        return
    if result.cache_hit:
        print(
            f"[GT_COST] q{result.query_idx} {result.doc_id}: "
            f"local_cache_hit total={total}"
        )
        return
    print(
        f"[GT_COST] q{result.query_idx} {result.doc_id}: "
        f"latency_ms={result.latency_ms:.3f} "
        f"input={result.input_tokens} "
        f"cached_input={result.cached_input_tokens} "
        f"output={result.output_tokens} "
        f"cost={_format_usd(result.cost_usd)} "
        f"run_input={input_tokens} "
        f"run_cached_input={cached_input_tokens} "
        f"run_output={output_tokens} "
        f"total={total}"
    )


def _print_usage_summary(
    results: Sequence[DocRunResult],
    *,
    run_latency_ms: float | None = None,
    log_path: Path | None = None,
) -> None:
    metrics = _run_metrics(results)
    if run_latency_ms is not None:
        print(f"Run Latency:  {run_latency_ms:.3f} ms")
    print(f"API Latency:  {metrics.api_latency_ms:.3f} ms")
    print(f"API Input:    {metrics.input_tokens}")
    print(f"API Cached:   {metrics.cached_input_tokens}")
    print(f"API Output:   {metrics.output_tokens}")
    print(f"API Cost:     {_format_usd(metrics.cost_usd)}")
    print(f"Local Cache:  {metrics.local_cache_hits}")
    if log_path is not None:
        print(f"Log File:     {log_path}")


def _print_summary(summary: GenerationSummary) -> None:
    print("=== Ground Truth Generation ===")
    print(f"Dataset Root: {summary.dataset_root}")
    print(f"Query:        q{summary.query.idx} ({summary.query.answer_type})")
    print(f"Selected:     {summary.selected_count}")
    print(f"Generated:    {summary.generated_count}")
    print(f"Skipped:      {summary.skipped_existing_count}")
    print(f"Failed:       {summary.failed_count}")
    _print_usage_summary(
        summary.results,
        run_latency_ms=summary.run_latency_ms,
        log_path=summary.log_path,
    )
    for result in summary.results:
        suffix = " cache_hit" if result.cache_hit else ""
        if result.status == "failed":
            print(f"  q{result.query_idx} {result.doc_id}: failed - {result.error}")
        else:
            print(f"  q{result.query_idx} {result.doc_id}: {result.status}{suffix}")


def _print_batch_summary(summary: BatchGenerationSummary) -> None:
    print("=== Ground Truth Generation ===")
    print(f"Dataset Root: {summary.dataset_root}")
    print(
        "Queries:      "
        + ", ".join(f"q{query.idx} ({query.answer_type})" for query in summary.queries)
    )
    print(f"Selected:     {summary.selected_count}")
    print(f"Generated:    {summary.generated_count}")
    print(f"Skipped:      {summary.skipped_existing_count}")
    print(f"Failed:       {summary.failed_count}")
    _print_usage_summary(
        summary.results,
        run_latency_ms=summary.run_latency_ms,
        log_path=summary.log_path,
    )
    for result in summary.results:
        suffix = " cache_hit" if result.cache_hit else ""
        if result.status == "failed":
            print(f"  q{result.query_idx} {result.doc_id}: failed - {result.error}")
        else:
            print(f"  q{result.query_idx} {result.doc_id}: {result.status}{suffix}")


if __name__ == "__main__":
    main()
