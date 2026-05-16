"""QAAgentExtractor - agentic document question-answering baseline.

This baseline gives an LLM agent document-reading and analysis tools and asks it
to answer one question for one document.
"""

from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import datetime
import decimal
import functools
import itertools
import json
import math
import multiprocessing
import operator
import os
import re
import statistics
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional analysis dependency
    pd = None  # type: ignore[assignment]

from agent.baselines.base import DocInputs, ExtractionResult
from agent.rule_runtime.context import _ensure_request_within_model_context
from agent.tool_agent.document import DocumentContext
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.cache import CachedLLMCaller

_DECODER = json.JSONDecoder()

_MAX_TURNS = 8
_MAX_AGENT_TOKENS = 1200
_MAX_ANSWER_TOKENS = 500
_OBSERVATION_MAX_CHARS = 8000
_HISTORY_KEEP_RECENT = 6

_SEARCH_TOP_K_DEFAULT = 5
_SEARCH_TOP_K_CAP = 10
_SEARCH_PREVIEW_CHARS = 200
_REGEX_TOP_K_DEFAULT = 20
_REGEX_TOP_K_CAP = 100
_REGEX_PREVIEW_CHARS = 300
_EMBED_TOP_K_DEFAULT = 5
_EMBED_TOP_K_CAP = 10
_EMBED_CACHE_DIR = Path(".cache") / "qa_agent_embeddings"
_WINDOW_CHUNK_SIZE = 2500
_WINDOW_CHUNK_STRIDE = 1800
_BM25_K1 = 1.5
_BM25_B = 0.75
_PYTHON_TIMEOUT_DEFAULT = 3
_PYTHON_TIMEOUT_CAP = 10
_PYTHON_CODE_MAX_CHARS = 20000
_PYTHON_OUTPUT_MAX_CHARS_DEFAULT = 4000
_PYTHON_OUTPUT_MAX_CHARS_CAP = 12000
_PYTHON_RESULT_ITEM_CAP = 100
_PYTHON_RESULT_STRING_CAP = 2000

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(slots=True)
class _QAToolResult:
    name: str
    success: bool
    data: Any
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass(slots=True)
class _QAToolPayload:
    data: Any
    cost_usd: float = 0.0


@dataclass(slots=True)
class _QAHistoryTurn:
    turn_index: int
    action_json: str
    tool_name: str | None
    observation: str


@dataclass(slots=True)
class _Chunk:
    chunk_id: str
    kind: str
    text: str
    start: int | None = None
    end: int | None = None
    section_id: int | None = None
    heading: str | None = None
    page_no: int | None = None


@dataclass(slots=True)
class _EmbeddingIndex:
    chunks: list[_Chunk]
    matrix: np.ndarray
    cache_path: Path
    build_cost_usd: float
    cache_hit: bool


class _QAToolRegistry:
    """Document-reading tools for the QA baseline."""

    TOOL_NAMES: tuple[str, ...] = (
        "semantic_search",
        "keyword_search",
        "regex_search",
        "read_chunk",
        "read_pages",
        "python",
    )

    def __init__(
        self,
        doc: DocumentContext,
        *,
        embedding_provider: str,
        embedding_model: str,
    ) -> None:
        self._doc = doc
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_index: _EmbeddingIndex | None = None
        self._query_embeddings: dict[str, np.ndarray] = {}
        self._chunks: list[_Chunk] | None = None
        self._chunk_by_id: dict[str, _Chunk] | None = None
        self._dispatch = {
            "semantic_search": self._tool_semantic_search,
            "keyword_search": self._tool_keyword_search,
            "regex_search": self._tool_regex_search,
            "read_chunk": self._tool_read_chunk,
            "read_pages": self._tool_read_pages,
            "python": self._tool_python,
        }

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> _QAToolResult:
        if tool_name not in self._dispatch:
            return _QAToolResult(
                name=tool_name,
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self.TOOL_NAMES)}",
            )
        t0 = time.perf_counter()
        try:
            data = self._dispatch[tool_name](args)
            if isinstance(data, _QAToolPayload):
                payload = data.data
                cost_usd = data.cost_usd
            else:
                payload = data
                cost_usd = 0.0
            return _QAToolResult(
                name=tool_name,
                success=True,
                data=payload,
                cost_usd=cost_usd,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            return _QAToolResult(
                name=tool_name,
                success=False,
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def _tool_semantic_search(self, args: dict[str, Any]) -> _QAToolPayload | list[dict[str, Any]]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return [{"error": "Missing or empty 'query'"}]
        top_k = _positive_int_arg(
            args,
            "top_k",
            default=_EMBED_TOP_K_DEFAULT,
            cap=_EMBED_TOP_K_CAP,
        )
        if not self._get_chunks():
            return []

        index = self._get_embedding_index()
        query_vec, query_cost = self._get_query_embedding(query)
        from core.embed.embeddings import cosine_sim_batch

        scores = cosine_sim_batch(index.matrix, query_vec)
        if scores.size == 0:
            return _QAToolPayload([], cost_usd=index.build_cost_usd + query_cost)

        order = np.argsort(-scores)[:top_k]
        hits = [
            _format_chunk_hit(index.chunks[int(idx)], float(scores[int(idx)]))
            for idx in order
        ]
        return _QAToolPayload(hits, cost_usd=index.build_cost_usd + query_cost)

    def _tool_keyword_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return [{"error": "Missing or empty 'query'"}]
        top_k = _positive_int_arg(
            args,
            "top_k",
            default=_SEARCH_TOP_K_DEFAULT,
            cap=_SEARCH_TOP_K_CAP,
        )

        chunks = self._get_chunks()
        query_terms = _tokenize(query)
        scored = _bm25_rank(chunks, query_terms)
        return [
            _format_chunk_hit(chunk, score)
            for score, chunk in scored[:top_k]
        ]

    def _tool_regex_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return [{"error": "Missing or empty 'pattern'"}]
        case_sensitive = bool(args.get("case_sensitive", False))
        top_k = _positive_int_arg(
            args,
            "top_k",
            default=_REGEX_TOP_K_DEFAULT,
            cap=_REGEX_TOP_K_CAP,
        )
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return [{"error": f"Invalid regex: {exc}"}]

        hits: list[dict[str, Any]] = []
        for chunk in self._get_chunks():
            for match in compiled.finditer(chunk.text):
                hit = _format_chunk_hit(chunk, 1.0)
                hit["match"] = match.group(0)
                preview, preview_start, preview_end = _char_window_preview_bounds(
                    chunk.text,
                    match.start(),
                    match.end(),
                    _REGEX_PREVIEW_CHARS,
                )
                hit["preview"] = preview
                hit["preview_chars"] = len(preview)
                hit["truncated_left_chars"] = preview_start
                hit["truncated_right_chars"] = max(0, len(chunk.text) - preview_end)
                hit["truncated_chars"] = (
                    hit["truncated_left_chars"] + hit["truncated_right_chars"]
                )
                hit["is_truncated"] = hit["truncated_chars"] > 0
                if hit["is_truncated"]:
                    hit["truncation_note"] = (
                        f"preview truncated; {hit['truncated_chars']} chars omitted"
                    )
                hits.append(hit)
                if len(hits) >= top_k:
                    return hits
        return hits

    def _tool_python(self, args: dict[str, Any]) -> str:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return "STDERR:\nMissing or empty 'code'"
        timeout_s = _positive_int_arg(
            args,
            "timeout_s",
            default=_PYTHON_TIMEOUT_DEFAULT,
            cap=_PYTHON_TIMEOUT_CAP,
        )
        max_chars = _positive_int_arg(
            args,
            "max_chars",
            default=_PYTHON_OUTPUT_MAX_CHARS_DEFAULT,
            cap=_PYTHON_OUTPUT_MAX_CHARS_CAP,
        )
        return _execute_python_tool(
            code=code,
            doc=self._doc,
            timeout_s=timeout_s,
            max_chars=max_chars,
        )

    def _tool_read_chunk(self, args: dict[str, Any]) -> str:
        chunk_id = args.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            return "ERROR: Missing or empty 'chunk_id'"
        chunk = self._get_chunk_by_id().get(chunk_id)
        if chunk is None:
            available = ", ".join(chunk.chunk_id for chunk in self._get_chunks()[:10])
            return f"ERROR: chunk_id={chunk_id!r} not found. Available examples: {available}"
        return chunk.text

    def _tool_read_pages(self, args: dict[str, Any]) -> str:
        start = args.get("start")
        end = args.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            return "ERROR: Missing or non-int 'start'/'end'"
        if start > end:
            return "ERROR: start must be <= end"
        if not self._doc.entries:
            return "ERROR: page metadata unavailable; use read_chunk."

        parts: list[str] = []
        for page_no in range(start, end + 1):
            page_text = _format_page_text(self._doc.entries, page_no)
            if page_text:
                parts.append(f"[page {page_no}]\n{page_text}")
        if not parts:
            return f"ERROR: no entries found for pages [{start}, {end}]"
        return "\n\n".join(parts)

    def _get_chunks(self) -> list[_Chunk]:
        if self._chunks is None:
            chunks = _section_chunks(self._doc)
            if not chunks:
                chunks = _window_chunks(self._doc.normalized_text)
            self._chunks = _assign_chunk_ids(chunks)
            self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        return self._chunks

    def _get_chunk_by_id(self) -> dict[str, _Chunk]:
        if self._chunk_by_id is None:
            self._get_chunks()
        return self._chunk_by_id or {}

    def _get_embedding_index(self) -> _EmbeddingIndex:
        if self._embedding_index is not None:
            return _EmbeddingIndex(
                chunks=self._embedding_index.chunks,
                matrix=self._embedding_index.matrix,
                cache_path=self._embedding_index.cache_path,
                build_cost_usd=0.0,
                cache_hit=True,
            )

        chunks = self._get_chunks()
        cache_path = _embedding_cache_path(
            doc_id=self._doc.doc_id,
            provider=self._embedding_provider,
            model=self._embedding_model,
            chunks=chunks,
        )
        meta_path = cache_path.with_suffix(".json")
        if cache_path.exists() and meta_path.exists():
            matrix = np.load(cache_path)["embeddings"].astype(np.float32)
            self._embedding_index = _EmbeddingIndex(
                chunks=chunks,
                matrix=matrix,
                cache_path=cache_path,
                build_cost_usd=0.0,
                cache_hit=True,
            )
            return self._embedding_index

        from core.embed.embeddings import get_embedding_cost, get_embeddings_batch

        before_tokens, before_cost = get_embedding_cost()
        del before_tokens
        texts = [_embedding_text(chunk) for chunk in chunks]
        vectors = get_embeddings_batch(
            texts,
            model=self._embedding_model,
            provider=self._embedding_provider,
            show_progress=False,
        )
        _after_tokens, after_cost = get_embedding_cost()
        build_cost = max(0.0, after_cost - before_cost)
        matrix = np.asarray(vectors, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=matrix)
        meta_path.write_text(
            json.dumps(
                {
                    "doc_hash": _doc_hash(self._doc.doc_id),
                    "provider": self._embedding_provider,
                    "model": self._embedding_model,
                    "chunk_count": len(chunks),
                    "chunks_hash": _chunks_hash(chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._embedding_index = _EmbeddingIndex(
            chunks=chunks,
            matrix=matrix,
            cache_path=cache_path,
            build_cost_usd=build_cost,
            cache_hit=False,
        )
        return self._embedding_index

    def _get_query_embedding(self, query: str) -> tuple[np.ndarray, float]:
        cache_key = f"{self._embedding_provider}|{self._embedding_model}|{query}"
        if cache_key in self._query_embeddings:
            return self._query_embeddings[cache_key], 0.0

        from core.embed.embeddings import get_embedding, get_embedding_cost

        _before_tokens, before_cost = get_embedding_cost()
        vector = get_embedding(
            query,
            model=self._embedding_model,
            provider=self._embedding_provider,
        )
        _after_tokens, after_cost = get_embedding_cost()
        cost = max(0.0, after_cost - before_cost)
        arr = np.asarray(vector, dtype=np.float32)
        self._query_embeddings[cache_key] = arr
        return arr, cost


class QAAgentExtractor:
    name: str = "qa-agent"

    def extract(
        self,
        *,
        query_idx: int,
        query_text: str,
        doc_id: str,
        doc_inputs: DocInputs,
        cached_caller: CachedLLMCaller,
        llm_provider: str = "azure",
        llm_model: str = "gpt-5.4-mini",
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        max_turns: int = _MAX_TURNS,
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0

        doc_context = _to_document_context(query_idx, doc_id, doc_inputs)
        resolved_embedding_provider = _resolve_embedding_provider(embedding_provider)
        resolved_embedding_model = _resolve_embedding_model(
            resolved_embedding_provider,
            embedding_model,
        )
        registry = _QAToolRegistry(
            doc_context,
            embedding_provider=resolved_embedding_provider,
            embedding_model=resolved_embedding_model,
        )
        system_prompt = _build_system_prompt(query_text, doc_context)
        action_schema = _build_action_schema()

        history: list[_QAHistoryTurn] = []
        trace_tool_calls: list[dict[str, Any]] = []
        trace_observations: list[dict[str, Any]] = []
        truncated_search_hits: dict[str, int] = {}
        has_full_evidence = False

        for turn_index in range(max_turns):
            prompt = _serialize_prompt(system_prompt, _compact_history(history))
            try:
                _ensure_request_within_model_context(
                    prompt_text=prompt,
                    max_output_tokens=_MAX_AGENT_TOKENS,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    stage_label=f"qa-agent q{query_idx} turn={turn_index}",
                )
            except RuntimeError as exc:
                return _finish(
                    answer="Information not found.",
                    trace={
                        "turns": turn_index,
                        "tool_calls": trace_tool_calls,
                        "observations": trace_observations,
                        "termination_reason": "context_limit",
                        "error": str(exc),
                        "doc_chars": len(doc_context.normalized_text),
                        "has_structured_entries": bool(doc_context.entries),
                        "embedding_provider": resolved_embedding_provider,
                        "embedding_model": resolved_embedding_model,
                    },
                    total_cost=total_cost,
                    t0=t0,
                )

            cache_result = cached_caller.call(
                prompt=prompt,
                llm_provider=llm_provider,
                max_tokens=_MAX_AGENT_TOKENS,
                model=llm_model,
                response_schema=action_schema,
            )
            call_cost = compute_cost(
                cache_result.input_tokens,
                cache_result.output_tokens,
                llm_provider,
                model=llm_model,
            )
            total_cost += call_cost

            action = _parse_action(cache_result.response)
            if action is None:
                observation = "ERROR: Invalid JSON response from QA agent."
                history.append(
                    _QAHistoryTurn(
                        turn_index=turn_index,
                        action_json=cache_result.response[:500],
                        tool_name=None,
                        observation=observation,
                    )
                )
                trace_observations.append(
                    {"turn": turn_index, "error": "invalid_json", "raw": cache_result.response}
                )
                continue

            action_type = action.get("action")
            if action_type == "final":
                answer = (action.get("answer") or "").strip()
                if not trace_tool_calls:
                    observation = "ERROR: You must call at least one document tool before final."
                    history.append(
                        _QAHistoryTurn(
                            turn_index=turn_index,
                            action_json=json.dumps(action, ensure_ascii=False),
                            tool_name=None,
                            observation=observation,
                        )
                    )
                    trace_observations.append(
                        {"turn": turn_index, "error": "final_before_tool"}
                    )
                    continue
                if truncated_search_hits and not has_full_evidence:
                    truncated_summary = ", ".join(
                        f"{chunk_id} ({chars} chars truncated)"
                        for chunk_id, chars in list(truncated_search_hits.items())[:5]
                    )
                    observation = (
                        "ERROR: Search results are locator previews, not full evidence. "
                        f"Truncated search hit(s): {truncated_summary}. "
                        "Call read_chunk on the relevant chunk_id, read_pages, or python before final."
                    )
                    history.append(
                        _QAHistoryTurn(
                            turn_index=turn_index,
                            action_json=json.dumps(action, ensure_ascii=False),
                            tool_name=None,
                            observation=observation,
                        )
                    )
                    trace_observations.append(
                        {
                            "turn": turn_index,
                            "error": "final_after_truncated_search_without_full_read",
                            "truncated_hits": dict(truncated_search_hits),
                        }
                    )
                    continue
                return _finish(
                    answer=answer or "Information not found.",
                    trace={
                        "turns": turn_index + 1,
                        "tool_calls": trace_tool_calls,
                        "observations": trace_observations,
                        "termination_reason": "final",
                        "doc_chars": len(doc_context.normalized_text),
                        "has_structured_entries": bool(doc_context.entries),
                        "embedding_provider": resolved_embedding_provider,
                        "embedding_model": resolved_embedding_model,
                    },
                    total_cost=total_cost,
                    t0=t0,
                )

            if action_type != "tool":
                observation = f"ERROR: Unknown action {action_type!r}; use 'tool' or 'final'."
                history.append(
                    _QAHistoryTurn(
                        turn_index=turn_index,
                        action_json=json.dumps(action, ensure_ascii=False),
                        tool_name=None,
                        observation=observation,
                    )
                )
                trace_observations.append(
                    {"turn": turn_index, "error": "unknown_action", "action": action_type}
                )
                continue

            tool_name = action.get("tool")
            if not isinstance(tool_name, str):
                tool_name = ""
            tool_args = _parse_tool_args(action.get("args"))
            tool_result = registry.dispatch(tool_name, tool_args)
            total_cost += tool_result.cost_usd
            if tool_result.success:
                if tool_name in {"read_chunk", "read_pages", "python"}:
                    has_full_evidence = True
                elif tool_name in {"semantic_search", "keyword_search", "regex_search"}:
                    for chunk_id, truncated_chars in _truncated_hit_summary(tool_result.data).items():
                        truncated_search_hits.setdefault(chunk_id, truncated_chars)
            if tool_result.success:
                obs_text = (
                    tool_result.data
                    if isinstance(tool_result.data, str)
                    else json.dumps(tool_result.data, ensure_ascii=False, default=str)
                )
            else:
                obs_text = f"ERROR: {tool_result.error}"
            obs_text = _truncate_text(obs_text, _OBSERVATION_MAX_CHARS)

            trace_tool_calls.append(
                {
                    "turn": turn_index,
                    "tool": tool_name,
                    "args": tool_args,
                    "success": tool_result.success,
                    "latency_ms": tool_result.latency_ms,
                    "cost_usd": tool_result.cost_usd,
                }
            )
            trace_observations.append(
                {
                    "turn": turn_index,
                    "tool": tool_name,
                    "observation": obs_text,
                    "success": tool_result.success,
                }
            )
            history.append(
                _QAHistoryTurn(
                    turn_index=turn_index,
                    action_json=json.dumps(action, ensure_ascii=False),
                    tool_name=tool_name,
                    observation=obs_text,
                )
            )

        if not trace_tool_calls:
            forced_result = registry.dispatch("keyword_search", {"query": query_text, "top_k": 5})
            if forced_result.success:
                forced_obs = (
                    forced_result.data
                    if isinstance(forced_result.data, str)
                    else json.dumps(forced_result.data, ensure_ascii=False, default=str)
                )
            else:
                forced_obs = f"ERROR: {forced_result.error}"
            forced_obs = _truncate_text(forced_obs, _OBSERVATION_MAX_CHARS)
            trace_tool_calls.append(
                {
                    "turn": max_turns,
                    "tool": "keyword_search",
                    "args": {"query": query_text, "top_k": 5},
                    "success": forced_result.success,
                    "latency_ms": forced_result.latency_ms,
                    "forced": True,
                }
            )
            trace_observations.append(
                {
                    "turn": max_turns,
                    "tool": "keyword_search",
                    "observation": forced_obs,
                    "success": forced_result.success,
                    "forced": True,
                }
            )
            history.append(
                _QAHistoryTurn(
                    turn_index=max_turns,
                    action_json=json.dumps(
                        {"action": "tool", "tool": "keyword_search", "args": {"query": query_text}},
                        ensure_ascii=False,
                    ),
                    tool_name="keyword_search",
                    observation=forced_obs,
                )
            )

        answer, synthesis_cost = _synthesize_final_answer(
            query_text=query_text,
            history=history,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        total_cost += synthesis_cost
        return _finish(
            answer=answer,
            trace={
                "turns": max_turns,
                "tool_calls": trace_tool_calls,
                "observations": trace_observations,
                "termination_reason": "max_turns_synthesis",
                "doc_chars": len(doc_context.normalized_text),
                "has_structured_entries": bool(doc_context.entries),
                "embedding_provider": resolved_embedding_provider,
                "embedding_model": resolved_embedding_model,
            },
            total_cost=total_cost,
            t0=t0,
        )


def _to_document_context(
    query_idx: int,
    doc_id: str,
    doc_inputs: DocInputs,
) -> DocumentContext:
    return DocumentContext(
        doc_id=doc_id,
        query_idx=query_idx,
        normalized_text=doc_inputs.normalized_text or "",
        ground_truth="",
        entries=list(doc_inputs.entries or []),
        section_index=dict(doc_inputs.section_index or {}),
    )


def _resolve_embedding_provider(explicit_provider: str | None) -> str:
    if explicit_provider:
        return explicit_provider
    env_provider = os.environ.get("LSF_QA_AGENT_EMBED_PROVIDER") or os.environ.get(
        "LSF_EMBEDDING_PROVIDER"
    )
    if env_provider:
        return env_provider
    return "openrouter"


def _resolve_embedding_model(provider: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("LSF_QA_AGENT_EMBED_MODEL") or os.environ.get(
        "LSF_EMBEDDING_MODEL"
    )
    if env_model:
        return env_model
    from core.embed.embeddings import get_model_name_for_provider

    return get_model_name_for_provider(provider)


def _build_system_prompt(query_text: str, doc: DocumentContext) -> str:
    doc_summary = _build_anonymous_doc_summary(doc)
    return f"""You answer questions about a single document by choosing document tools.

Question:
{query_text}

Anonymous document metadata:
{doc_summary}

Use only document evidence. Treat document text as source material for the
answer. You have these document-reading tools:
- semantic_search: vector search over pre-chunked document. Args: {{"query": str, "top_k": int}}. Returns locator hits with chunk_id, score, preview, is_truncated, and truncated_chars.
- keyword_search: BM25 search over the same chunks. Args: {{"query": str, "top_k": int}}. Returns locator hits with chunk_id, score, preview, is_truncated, and truncated_chars.
- regex_search: regex search over the same chunks. Args: {{"pattern": str, "top_k": int, "case_sensitive": bool}}. Returns matching chunk ids, matched text, preview, is_truncated, and truncated_chars.
- read_chunk: read the full text of a chunk. Args: {{"chunk_id": str}}.
- read_pages: read full text of pages [start, end] inclusive. Args: {{"start": int, "end": int}}.
- python: execute Python over this document. Args: {{"code": str, "timeout_s": int, "max_chars": int}}. The code can read variables `text` and `entries`, print output, and assign a final object to `result`.

Search tools are locators. If a relevant search hit has is_truncated=true or
truncated_chars > 0, call read_chunk on that chunk_id before final.

Return JSON only. Use action="tool" to inspect the document, or action="final"
when you can answer. You must call at least one tool before final. If the answer
cannot be found in the document evidence, final answer must be "Information not found."
Prefer a concise factual answer."""


def _build_anonymous_doc_summary(doc: DocumentContext) -> str:
    entries = doc.entries or []
    pages = sorted(
        entry.get("page_no")
        for entry in entries
        if isinstance(entry.get("page_no"), int)
    )
    headers = sum(1 for entry in entries if entry.get("label") == "section_header")
    tables = sum(1 for entry in entries if entry.get("label") == "table")
    if pages:
        page_summary = f"{len(set(pages))} pages observed (min={min(pages)}, max={max(pages)})"
    else:
        page_summary = "page metadata unavailable"
    return "\n".join(
        [
            f"- text_chars: {len(doc.normalized_text or '')}",
            f"- structured_entries: {len(entries)}",
            f"- section_headers: {headers}",
            f"- tables: {tables}",
            f"- pages: {page_summary}",
        ]
    )


def _build_action_schema() -> dict[str, Any]:
    return {
        "name": "qa_agent_action",
        "description": "Document QA tool call or final answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["tool", "final"]},
                "reasoning": {"type": "string"},
                "tool": {"type": ["string", "null"]},
                "args": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
            },
            "required": ["action", "reasoning", "tool", "args", "answer"],
        },
    }


def _serialize_prompt(system_prompt: str, history: list[_QAHistoryTurn]) -> str:
    parts = [f"<system>\n{system_prompt}\n</system>"]
    for turn in history:
        parts.append(
            f"\n<turn idx={turn.turn_index}>\n"
            f"Agent: {turn.action_json}\n"
            f"Observation: {turn.observation}\n"
            "</turn>"
        )
    parts.append(
        "\n\nDecide the next JSON action. For tool actions, put tool arguments "
        'as a JSON string in "args".'
    )
    return "\n".join(parts)


def _compact_history(history: list[_QAHistoryTurn]) -> list[_QAHistoryTurn]:
    if len(history) <= _HISTORY_KEEP_RECENT:
        return history
    compacted: list[_QAHistoryTurn] = []
    cutoff = len(history) - _HISTORY_KEEP_RECENT
    for turn in history[:cutoff]:
        compacted.append(
            _QAHistoryTurn(
                turn_index=turn.turn_index,
                action_json=turn.action_json,
                tool_name=turn.tool_name,
                observation="[summary] " + turn.observation[:500].replace("\n", " "),
            )
        )
    compacted.extend(history[cutoff:])
    return compacted


def _parse_action(raw_response: str) -> dict[str, Any] | None:
    try:
        action, _end = _DECODER.raw_decode(raw_response.lstrip())
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) else None


def _parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str) or not raw_args.strip():
        return {}
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _synthesize_final_answer(
    *,
    query_text: str,
    history: list[_QAHistoryTurn],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> tuple[str, float]:
    evidence_blocks = []
    for turn in history:
        if turn.tool_name:
            evidence_blocks.append(
                f"[turn={turn.turn_index} tool={turn.tool_name}]\n{turn.observation}"
            )
    evidence = _truncate_text("\n\n".join(evidence_blocks), 24000)
    prompt = f"""Using only the document evidence below, answer the question.

Question:
{query_text}

Evidence collected by tools:
{evidence}

Instructions:
- Answer with a concise factual phrase when possible.
- If the evidence does not contain the answer, say "Information not found."
- Do not use outside knowledge.

Answer:"""
    result = cached_caller.call(
        prompt=prompt,
        llm_provider=llm_provider,
        max_tokens=_MAX_ANSWER_TOKENS,
        model=llm_model,
    )
    cost = compute_cost(
        result.input_tokens,
        result.output_tokens,
        llm_provider,
        model=llm_model,
    )
    return _strip_final_prefix(result.response.strip()) or "Information not found.", cost


def _finish(
    *,
    answer: str,
    trace: dict[str, Any],
    total_cost: float,
    t0: float,
) -> ExtractionResult:
    return ExtractionResult(
        generated_answer=_strip_final_prefix(answer).strip() or "Information not found.",
        trace=trace,
        cost_usd=total_cost,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _strip_final_prefix(text: str) -> str:
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:", 1)[1].strip()
    return text


def _positive_int_arg(
    args: dict[str, Any],
    key: str,
    *,
    default: int,
    cap: int,
) -> int:
    value = args.get(key)
    if isinstance(value, int) and value > 0:
        return min(value, cap)
    return default


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total {len(text)} chars]"


def _truncated_hit_summary(data: Any) -> dict[str, int]:
    if not isinstance(data, list):
        return {}
    summary: dict[str, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        truncated_chars = item.get("truncated_chars", 0)
        if (
            isinstance(chunk_id, str)
            and isinstance(truncated_chars, int)
            and truncated_chars > 0
        ):
            summary[chunk_id] = truncated_chars
    return summary


def _tokenize(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_/%$.,-]*", text.casefold())
    return [term for term in terms if len(term) > 2 and term not in _STOPWORDS]


def _bm25_rank(chunks: list[_Chunk], query_terms: list[str]) -> list[tuple[float, _Chunk]]:
    if not chunks or not query_terms:
        return []

    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_docs = len(chunks)
    scored: list[tuple[float, _Chunk]] = []
    for chunk, tokens in zip(chunks, tokenized):
        if not tokens:
            continue
        counts = collections.Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * doc_len / max(avg_len, 1e-9))
            score += idf * (tf * (_BM25_K1 + 1.0) / denom)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return scored


def _format_chunk_hit(chunk: _Chunk, score: float) -> dict[str, Any]:
    preview = chunk.text[:_SEARCH_PREVIEW_CHARS]
    total_chars = len(chunk.text)
    preview_chars = len(preview)
    truncated_chars = max(0, total_chars - preview_chars)
    hit: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "score": round(float(score), 4),
        "preview": preview,
        "preview_chars": preview_chars,
        "total_chars": total_chars,
        "truncated_chars": truncated_chars,
        "is_truncated": truncated_chars > 0,
    }
    if truncated_chars > 0:
        hit["truncation_note"] = f"preview truncated; {truncated_chars} chars omitted"
    if chunk.page_no is not None:
        hit["page"] = chunk.page_no
    return hit


def _char_window_preview_bounds(
    text: str, start: int, end: int, max_chars: int
) -> tuple[str, int, int]:
    if max_chars <= 0:
        return "", start, start
    match_len = max(0, end - start)
    side = max(0, (max_chars - match_len) // 2)
    preview_start = max(0, start - side)
    preview_end = min(len(text), preview_start + max_chars)
    if preview_end - preview_start < max_chars:
        preview_start = max(0, preview_end - max_chars)
    prefix = "... " if preview_start > 0 else ""
    suffix = " ..." if preview_end < len(text) else ""
    return prefix + text[preview_start:preview_end] + suffix, preview_start, preview_end


def _char_window_preview(text: str, start: int, end: int, max_chars: int) -> str:
    preview, _preview_start, _preview_end = _char_window_preview_bounds(
        text, start, end, max_chars
    )
    return preview


def _section_chunks(doc: DocumentContext) -> list[_Chunk]:
    entries = doc.entries or []
    chunks: list[_Chunk] = []
    for idx, entry in enumerate(entries):
        if entry.get("label") != "section_header":
            continue
        ordered = sorted(_collect_descendants(idx, entries))
        parts = [(entry.get("text") or "").strip()]
        page_no = entry.get("page_no") if isinstance(entry.get("page_no"), int) else None
        for child_idx in ordered:
            if child_idx == idx:
                continue
            child = entries[child_idx]
            label = child.get("label", "")
            text = (child.get("text") or "").strip()
            if label == "section_header" and text:
                parts.append(f"\n## {text}\n")
            elif label == "list_item" and text:
                parts.append(f"- {text}")
            elif label == "table":
                parts.append(_format_table_marker(child_idx, child))
            elif text:
                parts.append(text)
            child_page = child.get("page_no")
            if page_no is None and isinstance(child_page, int):
                page_no = child_page
        chunk_text = "\n".join(part for part in parts if part).strip()
        if chunk_text:
            chunks.append(
                _Chunk(
                    chunk_id="",
                    kind="section",
                    text=chunk_text,
                    section_id=idx,
                    heading=(entry.get("text") or "").strip(),
                    page_no=page_no,
                )
            )
    return chunks


def _window_chunks(text: str) -> list[_Chunk]:
    text = text or ""
    if not text:
        return []
    chunk_size = _WINDOW_CHUNK_SIZE
    stride = _WINDOW_CHUNK_STRIDE
    chunks = []
    for start in range(0, len(text), stride):
        end = min(len(text), start + chunk_size)
        chunks.append(_Chunk(chunk_id="", kind="window", text=text[start:end], start=start, end=end))
        if end == len(text):
            break
    return chunks


def _assign_chunk_ids(chunks: list[_Chunk]) -> list[_Chunk]:
    page_counts: dict[int, int] = {}
    assigned: list[_Chunk] = []
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk.page_no, int):
            page_idx = page_counts.get(chunk.page_no, 0)
            page_counts[chunk.page_no] = page_idx + 1
            chunk_id = f"p{chunk.page_no}_c{page_idx}"
        else:
            chunk_id = f"c{idx}"
        assigned.append(
            _Chunk(
                chunk_id=chunk_id,
                kind=chunk.kind,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                section_id=chunk.section_id,
                heading=chunk.heading,
                page_no=chunk.page_no,
            )
        )
    return assigned


def _embedding_text(chunk: _Chunk) -> str:
    labels = []
    if chunk.heading:
        labels.append(f"Heading: {chunk.heading}")
    if chunk.page_no is not None:
        labels.append(f"Page: {chunk.page_no}")
    if chunk.section_id is not None:
        labels.append(f"Section id: {chunk.section_id}")
    prefix = "\n".join(labels)
    if prefix:
        return f"{prefix}\n\n{chunk.text}"
    return chunk.text


def _chunks_hash(chunks: list[_Chunk]) -> str:
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "kind": chunk.kind,
            "start": chunk.start,
            "end": chunk.end,
            "section_id": chunk.section_id,
            "heading": chunk.heading,
            "page_no": chunk.page_no,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _embedding_cache_path(
    *,
    doc_id: str,
    provider: str,
    model: str,
    chunks: list[_Chunk],
) -> Path:
    doc_hash = _doc_hash(doc_id)
    identity = json.dumps(
        {
            "doc_hash": doc_hash,
            "provider": provider,
            "model": model,
            "chunks_hash": _chunks_hash(chunks),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = sha256(identity.encode("utf-8")).hexdigest()[:24]
    return _EMBED_CACHE_DIR / provider / f"doc-{doc_hash}-{digest}.npz"


def _doc_hash(doc_id: str) -> str:
    return sha256(doc_id.encode("utf-8")).hexdigest()[:16]


_PYTHON_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "collections",
        "datetime",
        "decimal",
        "functools",
        "itertools",
        "json",
        "math",
        "numpy",
        "operator",
        "pandas",
        "re",
        "statistics",
    }
)
_PYTHON_SAFE_MODULES: dict[str, Any] = {
    "collections": collections,
    "datetime": datetime,
    "decimal": decimal,
    "functools": functools,
    "itertools": itertools,
    "json": json,
    "math": math,
    "np": np,
    "numpy": np,
    "operator": operator,
    "re": re,
    "statistics": statistics,
}
if pd is not None:
    _PYTHON_SAFE_MODULES["pd"] = pd
    _PYTHON_SAFE_MODULES["pandas"] = pd
_PYTHON_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)


def _execute_python_tool(
    *,
    code: str,
    doc: DocumentContext,
    timeout_s: int,
    max_chars: int,
) -> str:
    if len(code) > _PYTHON_CODE_MAX_CHARS:
        return f"STDERR:\ncode too long: {len(code)} chars > {_PYTHON_CODE_MAX_CHARS}"

    violations = _validate_python_tool_ast(code)
    if violations:
        return "STDERR:\nAST violations: " + "; ".join(violations[:10])

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_python_tool_in_subprocess,
        args=(code, doc.normalized_text or "", list(doc.entries or []), child_conn, max_chars),
    )
    t0 = time.perf_counter()
    child_conn_open = True
    payload: dict[str, Any] | None = None
    try:
        process.start()
        child_conn.close()
        child_conn_open = False
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            remaining_s = deadline - time.perf_counter()
            if parent_conn.poll(min(0.05, max(0.0, remaining_s))):
                payload = parent_conn.recv()
                break
            if not process.is_alive():
                break

        if payload is None and process.is_alive():
            process.terminate()
            process.join()
            return f"STDERR:\nTimeoutError: execution exceeded {timeout_s}s"

        process.join()
        if payload is None:
            if parent_conn.poll():
                payload = parent_conn.recv()
            else:
                payload = {
                    "ok": False,
                    "error": f"sandbox process exited with code {process.exitcode}",
                }
    except Exception as exc:
        return f"STDERR:\n{type(exc).__name__}: {exc}"
    finally:
        if child_conn_open:
            child_conn.close()
        parent_conn.close()

    exec_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return _format_python_payload(payload, max_chars=max_chars, exec_time_ms=exec_time_ms)


def _validate_python_tool_ast(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _PYTHON_ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _PYTHON_ALLOWED_IMPORTS:
                violations.append(f"forbidden import from: {node.module or ''}")
        elif isinstance(node, ast.Name) and node.id in _PYTHON_FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"forbidden attribute: {node.attr}")
    return violations


class _CappedStdout:
    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._parts: list[str] = []
        self._chars_seen = 0
        self.truncated = False

    def write(self, value: object) -> int:
        text = str(value)
        self._chars_seen += len(text)
        remaining = self._max_chars - sum(len(part) for part in self._parts)
        if remaining > 0:
            self._parts.append(text[:remaining])
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        text = "".join(self._parts)
        if self.truncated:
            return text + f"\n...[stdout truncated, total {self._chars_seen} chars]"
        return text


def _run_python_tool_in_subprocess(
    code: str,
    text: str,
    entries: list[dict[str, Any]],
    child_conn: Any,
    max_chars: int,
) -> None:
    try:
        namespace = _build_python_tool_globals()
        namespace.update(
            {
                "text": text,
                "entries": entries,
            }
        )
        stdout = _CappedStdout(max_chars)
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)  # noqa: S102 - intentional restricted tool execution
        child_conn.send(
            {
                "ok": True,
                "stdout": stdout.getvalue(),
                "result_set": "result" in namespace,
                "result": _make_python_result_safe(namespace.get("result")),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        child_conn.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        child_conn.close()


def _build_python_tool_globals() -> dict[str, Any]:
    safe_builtins: dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "StopIteration": StopIteration,
    }
    real_import = builtins.__import__

    def safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".", 1)[0]
        if root in _PYTHON_ALLOWED_IMPORTS:
            return real_import(name, *args, **kwargs)
        raise ImportError(f"import of '{name}' is not allowed")

    safe_builtins["__import__"] = safe_import
    return {"__builtins__": safe_builtins, **_PYTHON_SAFE_MODULES}


def _make_python_result_safe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _PYTHON_RESULT_STRING_CAP:
            return _truncate_text(value, _PYTHON_RESULT_STRING_CAP)
        return value
    if isinstance(value, (list, tuple)):
        return [
            _make_python_result_safe(item, depth + 1)
            for item in list(value)[:_PYTHON_RESULT_ITEM_CAP]
        ]
    if isinstance(value, set):
        items = sorted((repr(item) for item in value))[:_PYTHON_RESULT_ITEM_CAP]
        return items
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _PYTHON_RESULT_ITEM_CAP:
                break
            safe[str(key)] = _make_python_result_safe(item, depth + 1)
        return safe
    return repr(value)


def _cap_python_result(value: Any, max_chars: int) -> Any:
    raw = json.dumps(value, ensure_ascii=False, default=repr)
    if len(raw) <= max_chars:
        return value
    return _truncate_text(raw, max_chars)


def _format_python_payload(
    payload: dict[str, Any],
    *,
    max_chars: int,
    exec_time_ms: float,
) -> str:
    if not payload.get("ok"):
        return f"STDERR:\n{payload.get('error') or 'unknown error'}"

    parts: list[str] = []
    stdout = str(payload.get("stdout") or "")
    if stdout:
        parts.append("STDOUT:\n" + _truncate_text(stdout, max_chars))
    if payload.get("result_set"):
        result = _cap_python_result(payload.get("result"), max_chars)
        result_text = json.dumps(result, ensure_ascii=False, default=repr, indent=2)
        parts.append("RESULT:\n" + _truncate_text(result_text, max_chars))
    parts.append(f"EXEC_TIME_MS: {exec_time_ms}")
    return "\n\n".join(parts)


def _collect_descendants(root_id: int, entries: list[dict[str, Any]]) -> set[int]:
    descendants: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        next_frontier: list[int] = []
        for pid in frontier:
            for idx, entry in enumerate(entries):
                if idx in descendants:
                    continue
                if entry.get("structure", {}).get("parent_id") == pid:
                    descendants.add(idx)
                    next_frontier.append(idx)
        frontier = next_frontier
    return descendants


def _format_table_marker(idx: int, entry: dict[str, Any]) -> str:
    table_data = entry.get("table_data") or {}
    rows = table_data.get("num_rows", "?")
    cols = table_data.get("num_cols", "?")
    page = entry.get("page_no", "?")
    marker = f"[Table id={idx} at p{page}, {rows}x{cols} cells]"
    table_text = (entry.get("text") or "").strip()
    if table_text:
        return f"{marker}\n{table_text}"
    return marker


def _format_page_text(entries: list[dict[str, Any]], page_no: int) -> str:
    body_parts: list[str] = []
    for idx, entry in enumerate(entries):
        if entry.get("page_no") != page_no:
            continue
        label = entry.get("label", "")
        text = (entry.get("text") or "").strip()
        if label == "section_header" and text:
            body_parts.append(f"\n## [id={idx}] {text}\n")
        elif label == "list_item" and text:
            body_parts.append(f"- {text}")
        elif label == "table":
            body_parts.append(_format_table_marker(idx, entry))
        elif text:
            body_parts.append(text)
    return "\n".join(body_parts).strip()
