"""Ground-truth generator for one-query, one-document PDF evaluation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
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
DEFAULT_INPUT_MODE = "auto"
DEFAULT_GENERATION_MODE = "single"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_CLAUDE_TIMEOUT_SEC = 600

InputMode = Literal["auto", "native-pdf", "text", "claude-read-pdf"]
ResolvedInputMode = Literal["native-pdf", "text", "claude-read-pdf"]
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
    results: tuple[DocRunResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AzureResponsesCallResult:
    """Raw Azure Responses API output plus provider-reported usage."""

    response: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float


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
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    progress_cost: bool = False,
) -> GenerationSummary:
    """Generate ground-truth answers for sampled PDFs.

    Args:
        target_dir: Dataset directory, either datasets/<name> or datasets/<name>/latest.
        query_idx: 1-based query index from queries.json.
        num_doc: Number of PDFs to sample before skipping existing GT; None means all.
        llm_provider: LLM provider. Native PDF mode currently supports azure.
        model: Model identifier, default gpt-5.4-mini.
        seed: Random sampling seed.
        input_mode: auto chooses provider default; native-pdf uses Azure Responses
            input_file; text uses extracted text; claude-read-pdf asks Claude Code
            to read the local PDF path.
        generation_mode: single answers this query with the single-answer schema;
            all answers the selected query set with the multi-answer schema.
        cache_db: SQLite LLM cache path.
        max_tokens: Max output tokens.
        claude_timeout_sec: Timeout for each Claude Code CLI call.
        progress_cost: Print per-call API usage/cost as the run progresses.
    """
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    resolved_generation_mode = resolve_generation_mode(generation_mode)
    dataset_root = resolve_dataset_root(target_dir)
    query = load_query(dataset_root / "queries.json", query_idx)
    pdf_paths = sample_pdf_paths(dataset_root / "raw", num_doc=num_doc, seed=seed)
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    native_caller = (
        NativePDFCacheCaller(cache_db) if resolved_input_mode == "native-pdf" else None
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

        try:
            if resolved_generation_mode == "all":
                response = _call_model_for_queries_for_doc(
                    pdf_path=pdf_path,
                    queries=[query],
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    native_caller=native_caller,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    claude_timeout_sec=claude_timeout_sec,
                )
                answer = parse_answers_response(response.response, [query])[query.idx]
                api_call_id = _build_result_api_call_id(
                    pdf_path=pdf_path,
                    queries=[query],
                    generation_mode=resolved_generation_mode,
                )
            else:
                response = _call_model_for_doc(
                    pdf_path=pdf_path,
                    query=query,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    native_caller=native_caller,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    claude_timeout_sec=claude_timeout_sec,
                )
                answer = parse_answer_response(response.response)
                api_call_id = ""
            _merge_ground_truth(output_path, gt_key, answer)
            _record_result(
                results,
                DocRunResult(
                    doc_id=doc_id,
                    output_path=output_path,
                    status="generated",
                    query_idx=query.idx,
                    cache_hit=response.cache_hit,
                    answer=answer,
                    input_tokens=response.input_tokens,
                    cached_input_tokens=response.cached_input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    api_call_id=api_call_id,
                ),
                progress_cost=progress_cost,
            )
        except Exception as exc:  # keep the batch moving across bad PDFs/API failures
            _record_result(
                results,
                DocRunResult(
                    doc_id=doc_id,
                    output_path=output_path,
                    status="failed",
                    query_idx=query.idx,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                progress_cost=progress_cost,
            )

    return GenerationSummary(
        dataset_root=dataset_root,
        query=query,
        selected_count=len(pdf_paths),
        generated_count=sum(r.status == "generated" for r in results),
        skipped_existing_count=sum(r.status == "skipped_existing" for r in results),
        failed_count=sum(r.status == "failed" for r in results),
        results=tuple(results),
    )


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
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    progress_cost: bool = False,
) -> BatchGenerationSummary:
    """Generate ground-truth answers for multiple queries using doc-major order.

    This is the preferred entry point when running more than one query. It keeps
    the same document adjacent across query calls, which gives provider-side
    prompt caching the best chance to reuse the long document prefix.

    Args:
        progress_cost: Print per-call API usage/cost as the run progresses.
    """
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    resolved_generation_mode = resolve_generation_mode(generation_mode)
    dataset_root = resolve_dataset_root(target_dir)
    queries = load_queries(dataset_root / "queries.json", query_indices)
    pdf_paths = sample_pdf_paths(dataset_root / "raw", num_doc=num_doc, seed=seed)
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    native_caller = (
        NativePDFCacheCaller(cache_db) if resolved_input_mode == "native-pdf" else None
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
            try:
                response = _call_model_for_queries_for_doc(
                    pdf_path=pdf_path,
                    queries=pending_queries,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    native_caller=native_caller,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    claude_timeout_sec=claude_timeout_sec,
                    document_text=document_text,
                )
                answers = parse_answers_response(response.response, pending_queries)
                _merge_ground_truth_answers(output_path, answers)
                for query in pending_queries:
                    _record_result(
                        results,
                        DocRunResult(
                            doc_id=doc_id,
                            output_path=output_path,
                            status="generated",
                            query_idx=query.idx,
                            cache_hit=response.cache_hit,
                            answer=answers[query.idx],
                            input_tokens=response.input_tokens,
                            cached_input_tokens=response.cached_input_tokens,
                            output_tokens=response.output_tokens,
                            cost_usd=response.cost_usd,
                            api_call_id=api_call_id,
                        ),
                        progress_cost=progress_cost,
                    )
            except Exception as exc:  # keep the batch moving across bad PDFs/API failures
                for query in pending_queries:
                    _record_result(
                        results,
                        DocRunResult(
                            doc_id=doc_id,
                            output_path=output_path,
                            status="failed",
                            query_idx=query.idx,
                            api_call_id=api_call_id,
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                        progress_cost=progress_cost,
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

            try:
                response = _call_model_for_doc(
                    pdf_path=pdf_path,
                    query=query,
                    llm_provider=resolved_provider,
                    model=resolved_model,
                    input_mode=resolved_input_mode,
                    native_caller=native_caller,
                    text_caller=text_caller,
                    claude_caller=claude_caller,
                    max_tokens=max_tokens,
                    claude_timeout_sec=claude_timeout_sec,
                    document_text=document_text,
                )
                answer = parse_answer_response(response.response)
                _merge_ground_truth(output_path, gt_key, answer)
                _record_result(
                    results,
                    DocRunResult(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="generated",
                        query_idx=query.idx,
                        cache_hit=response.cache_hit,
                        answer=answer,
                        input_tokens=response.input_tokens,
                        cached_input_tokens=response.cached_input_tokens,
                        output_tokens=response.output_tokens,
                        cost_usd=response.cost_usd,
                    ),
                    progress_cost=progress_cost,
                )
            except Exception as exc:  # keep the batch moving across bad PDFs/API failures
                _record_result(
                    results,
                    DocRunResult(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="failed",
                        query_idx=query.idx,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    progress_cost=progress_cost,
                )

    return BatchGenerationSummary(
        dataset_root=dataset_root,
        queries=tuple(queries),
        selected_count=len(pdf_paths),
        generated_count=sum(r.status == "generated" for r in results),
        skipped_existing_count=sum(r.status == "skipped_existing" for r in results),
        failed_count=sum(r.status == "failed" for r in results),
        results=tuple(results),
    )


class NativePDFCacheCaller:
    """Azure Responses API caller with the shared SQLite LLM cache."""

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
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> CacheResult:
        if llm_provider != "azure":
            raise ValueError(
                "native-pdf input mode currently supports only llm_provider='azure'"
            )
        resolved_model = _require_model(model)
        normalized_temperature = float(temperature)
        pdf_hash = _sha256_file(pdf_path)
        cache_key = _build_native_pdf_cache_key(
            pdf_hash=pdf_hash,
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
        call_result = _azure_responses_pdf_call(
            pdf_path=pdf_path,
            prompt=prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=normalized_temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if normalized_temperature == 0.0:
            metadata = json.dumps(
                {
                    "mode": "gt_gen_native_pdf_v1",
                    "pdf_name": pdf_path.name,
                    "pdf_sha256": pdf_hash,
                    "prompt": prompt,
                    "response_schema": response_schema,
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
        return "native-pdf" if resolved_provider == "azure" else "text"
    if mode == "native-pdf":
        if resolved_provider != "azure":
            raise ValueError("input_mode='native-pdf' is supported only for azure")
        return "native-pdf"
    if mode == "text":
        return "text"
    if mode == "claude-read-pdf":
        if resolved_provider != "claude-code":
            raise ValueError(
                "input_mode='claude-read-pdf' is supported only for claude-code"
            )
        return "claude-read-pdf"
    raise ValueError("input_mode must be one of: auto, native-pdf, text, claude-read-pdf")


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
    native_caller: NativePDFCacheCaller | None,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    document_text: str | None = None,
) -> CacheResult:
    prompt = build_ground_truth_prompt(query=query, doc_id=pdf_path.stem)
    response_schema = _answer_response_schema()
    if input_mode == "native-pdf":
        if native_caller is None:
            raise RuntimeError("native PDF caller is not initialized")
        return native_caller.call(
            pdf_path=pdf_path,
            prompt=prompt,
            llm_provider=llm_provider,
            model=model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=0,
        )
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
            temperature=0,
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
    native_caller: NativePDFCacheCaller | None,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    document_text: str | None = None,
) -> CacheResult:
    prompt = build_ground_truth_all_prompt(queries=queries, doc_id=pdf_path.stem)
    response_schema = _answers_response_schema(queries)
    if input_mode == "native-pdf":
        if native_caller is None:
            raise RuntimeError("native PDF caller is not initialized")
        return native_caller.call(
            pdf_path=pdf_path,
            prompt=prompt,
            llm_provider=llm_provider,
            model=model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            temperature=0,
        )
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
            temperature=0,
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


def build_ground_truth_prompt(*, query: QuerySpec, doc_id: str) -> str:
    return (
        _ground_truth_instructions()
        + "\n\n"
        f"Document id: {doc_id}\n"
        f"Question index: {query.idx}\n"
        f"Question: {query.text}\n"
        f"Answer type: {query.answer_type}\n"
    )


def build_ground_truth_text_prompt(
    *, query: QuerySpec, doc_id: str, document_text: str
) -> str:
    """Build a cache-friendly text prompt with document before query details."""
    return (
        _ground_truth_instructions()
        + "\n\n"
        f"Document id: {doc_id}\n\n"
        + document_text
        + "\n\n"
        + "[QUESTION]\n"
        f"Question index: {query.idx}\n"
        f"Question: {query.text}\n"
        f"Answer type: {query.answer_type}\n"
    )


def build_ground_truth_all_prompt(*, queries: Sequence[QuerySpec], doc_id: str) -> str:
    return (
        _ground_truth_all_instructions()
        + "\n\n"
        f"Document id: {doc_id}\n"
        + _format_query_list(queries)
    )


def build_ground_truth_all_text_prompt(
    *, queries: Sequence[QuerySpec], doc_id: str, document_text: str
) -> str:
    """Build a cache-friendly multi-query text prompt with document before queries."""
    return (
        _ground_truth_all_instructions()
        + "\n\n"
        f"Document id: {doc_id}\n\n"
        + document_text
        + "\n\n"
        + "[QUESTIONS]\n"
        + _format_query_list(queries)
    )


def _ground_truth_instructions() -> str:
    return (
        "You generate gold ground-truth answers for a PDF question-answering benchmark.\n"
        "Use only the attached PDF/document content. Do not use outside knowledge.\n"
        "Read the entire document before answering. Preserve names, docket numbers, dates, "
        "statutes, and numeric values exactly as presented when possible.\n"
        "If the answer is a list, return every distinct answer in document order. "
        "If the requested information is absent, use an explicit string required by the "
        "question such as \"not disclosed\" or \"not applicable\"; otherwise use \"None\".\n"
        "Return exactly one valid JSON object and no markdown:\n"
        "{\"answer\": <answer matching answer_type>, \"support\": \"short evidence quote or page note\"}"
    )


def _ground_truth_all_instructions() -> str:
    return (
        "You generate gold ground-truth answers for a PDF question-answering benchmark.\n"
        "Use only the attached PDF/document content. Do not use outside knowledge.\n"
        "Read the entire document before answering. Preserve names, docket numbers, dates, "
        "statutes, and numeric values exactly as presented when possible.\n"
        "Answer every listed question exactly once. If an answer is a list, return every "
        "distinct answer in document order. If requested information is absent, use an "
        "explicit string required by the question such as \"not disclosed\" or "
        "\"not applicable\"; otherwise use \"None\"."
    )


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


def _azure_responses_pdf_call(
    *,
    pdf_path: Path,
    prompt: str,
    model: str,
    max_tokens: int,
    response_schema: dict[str, Any] | None,
    temperature: float,
) -> AzureResponsesCallResult:
    client, deployment = _azure_responses_client_and_deployment(model)
    file_data = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    response = client.responses.create(
        model=deployment,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": pdf_path.name,
                        "file_data": f"data:application/pdf;base64,{file_data}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
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
            "answer": {},
            "support": {"type": "string"},
        },
        "required": ["answer", "support"],
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
                        "answer": {},
                        "support": {"type": "string"},
                    },
                    "required": ["query_idx", "answer", "support"],
                },
            }
        },
        "required": ["answers"],
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


def _build_native_pdf_cache_key(
    *,
    pdf_hash: str,
    prompt: str,
    llm_provider: str,
    model: str,
    max_tokens: int,
    response_schema: dict[str, Any] | None,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "mode": "gt_gen_native_pdf_v1",
            "pdf_sha256": pdf_hash,
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
        choices=["auto", "native-pdf", "text", "claude-read-pdf"],
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
        "--claude-timeout-sec",
        type=int,
        default=DEFAULT_CLAUDE_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--progress-cost",
        action="store_true",
        help="Print per-call API token usage and cumulative cost while running.",
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
            claude_timeout_sec=args.claude_timeout_sec,
            progress_cost=args.progress_cost,
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
        claude_timeout_sec=args.claude_timeout_sec,
        progress_cost=args.progress_cost,
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


def _record_result(
    results: list[DocRunResult],
    result: DocRunResult,
    *,
    progress_cost: bool,
) -> None:
    results.append(result)
    if progress_cost:
        _print_progress_cost(result, results)


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
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    local_cache_hits = 0
    seen_usage_keys: set[str] = set()
    for position, result in enumerate(results):
        if result.status != "generated":
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
    return input_tokens, cached_input_tokens, output_tokens, cost_usd, local_cache_hits


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
        f"input={result.input_tokens} "
        f"cached_input={result.cached_input_tokens} "
        f"output={result.output_tokens} "
        f"cost={_format_usd(result.cost_usd)} "
        f"run_input={input_tokens} "
        f"run_cached_input={cached_input_tokens} "
        f"run_output={output_tokens} "
        f"total={total}"
    )


def _print_usage_summary(results: Sequence[DocRunResult]) -> None:
    input_tokens, cached_input_tokens, output_tokens, cost_usd, local_cache_hits = (
        _api_usage_totals(results)
    )
    print(f"API Input:    {input_tokens}")
    print(f"API Cached:   {cached_input_tokens}")
    print(f"API Output:   {output_tokens}")
    print(f"API Cost:     {_format_usd(cost_usd)}")
    print(f"Local Cache:  {local_cache_hits}")


def _print_summary(summary: GenerationSummary) -> None:
    print("=== Ground Truth Generation ===")
    print(f"Dataset Root: {summary.dataset_root}")
    print(f"Query:        q{summary.query.idx} ({summary.query.answer_type})")
    print(f"Selected:     {summary.selected_count}")
    print(f"Generated:    {summary.generated_count}")
    print(f"Skipped:      {summary.skipped_existing_count}")
    print(f"Failed:       {summary.failed_count}")
    _print_usage_summary(summary.results)
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
    _print_usage_summary(summary.results)
    for result in summary.results:
        suffix = " cache_hit" if result.cache_hit else ""
        if result.status == "failed":
            print(f"  q{result.query_idx} {result.doc_id}: failed - {result.error}")
        else:
            print(f"  q{result.query_idx} {result.doc_id}: {result.status}{suffix}")


if __name__ == "__main__":
    main()
