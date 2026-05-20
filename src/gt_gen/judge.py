"""LLM judge for selecting the best ground-truth answer from multiple candidates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from core.pipeline.e2e_utils.cache import CacheResult, DEFAULT_CACHE_DB_PATH

from gt_gen.generator import (
    DEFAULT_CLAUDE_TIMEOUT_SEC,
    DEFAULT_INPUT_MODE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AzureResponsesTextCacheCaller,
    ClaudeCodeCacheCaller,
    InputMode,
    QuerySpec,
    ResolvedInputMode,
    _answer_value_schema,
    _atomic_write_json,
    _elapsed_ms,
    _extract_pdf_text_for_prompt,
    _format_usd,
    _load_gt_prompt_template,
    _make_run_logger,
    _parse_query_indices_arg,
    _render_gt_prompt,
    load_queries,
    normalize_llm_provider,
    resolve_dataset_root,
    resolve_input_mode,
    resolve_model_for_provider,
)

DEFAULT_JUDGE_MODE: "JudgeMode" = "per-query"
JudgeMode = Literal["per-query", "batched"]


@dataclass(frozen=True)
class Candidate:
    """One candidate answer for a query, sourced from any upstream pipeline."""

    source: str
    answer: Any
    reasoning: str = ""
    support: str = ""


@dataclass(frozen=True)
class JudgedAnswer:
    """Judge's pick for one query."""

    query_idx: int
    picked_index: int
    picked_source: str
    answer: Any
    reasoning: str


@dataclass(frozen=True)
class JudgeDocResult:
    """Per-(doc, query) outcome row used for the run summary and logger."""

    doc_id: str
    output_path: Path
    status: str
    query_idx: int = 0
    cache_hit: bool = False
    answer: Any = None
    picked_index: int = 0
    picked_source: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    api_call_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class JudgeSummary:
    """Run summary returned by judge_ground_truth_from_manifest()."""

    dataset_root: Path
    queries: tuple[QuerySpec, ...]
    selected_count: int
    judged_count: int
    skipped_existing_count: int
    skipped_no_candidates_count: int
    failed_count: int
    run_latency_ms: float = 0.0
    log_path: Path | None = None
    results: tuple[JudgeDocResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ManifestDocument:
    """One document entry parsed from the candidates manifest."""

    doc_id: str
    pdf_path: Path
    candidates: dict[int, tuple[Candidate, ...]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_document(
    *,
    pdf_path: Path,
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
    judge_mode: JudgeMode = DEFAULT_JUDGE_MODE,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_MODEL,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
    cache_db: str = DEFAULT_CACHE_DB_PATH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    text_caller: AzureResponsesTextCacheCaller | None = None,
    claude_caller: ClaudeCodeCacheCaller | None = None,
    document_text: str | None = None,
) -> tuple[list[JudgedAnswer], list[CacheResult]]:
    """Judge candidate answers for one document.

    Returns (judged_answers, cache_results). cache_results is one entry for
    judge_mode='batched' and one-per-query for 'per-query'.
    """
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    text_caller, claude_caller = _ensure_callers(
        resolved_provider, resolved_input_mode, cache_db, text_caller, claude_caller
    )

    if judge_mode == "batched":
        cache_result = _call_judge_for_queries_for_doc(
            pdf_path=pdf_path,
            queries=queries,
            candidates_by_query=candidates_by_query,
            llm_provider=resolved_provider,
            model=resolved_model,
            input_mode=resolved_input_mode,
            text_caller=text_caller,
            claude_caller=claude_caller,
            max_tokens=max_tokens,
            claude_timeout_sec=claude_timeout_sec,
            document_text=document_text,
        )
        parsed = parse_judges_response(
            cache_result.response, queries, candidates_by_query
        )
        judged = [
            JudgedAnswer(
                query_idx=q.idx,
                picked_index=parsed[q.idx][0],
                picked_source=parsed[q.idx][1],
                answer=parsed[q.idx][2],
                reasoning=parsed[q.idx][3],
            )
            for q in queries
        ]
        return judged, [cache_result]

    judged: list[JudgedAnswer] = []
    cache_results: list[CacheResult] = []
    for query in queries:
        candidates = list(candidates_by_query.get(query.idx, []))
        cache_result = _call_judge_for_query_for_doc(
            pdf_path=pdf_path,
            query=query,
            candidates=candidates,
            llm_provider=resolved_provider,
            model=resolved_model,
            input_mode=resolved_input_mode,
            text_caller=text_caller,
            claude_caller=claude_caller,
            max_tokens=max_tokens,
            claude_timeout_sec=claude_timeout_sec,
            document_text=document_text,
        )
        picked_index, picked_source, answer, reasoning = parse_judge_response(
            cache_result.response, candidates
        )
        judged.append(
            JudgedAnswer(
                query_idx=query.idx,
                picked_index=picked_index,
                picked_source=picked_source,
                answer=answer,
                reasoning=reasoning,
            )
        )
        cache_results.append(cache_result)
    return judged, cache_results


def judge_ground_truth_from_manifest(
    target_dir: str | Path,
    manifest_path: str | Path,
    query_indices: Sequence[int] | None = None,
    *,
    judge_mode: JudgeMode = DEFAULT_JUDGE_MODE,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_MODEL,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
    cache_db: str = DEFAULT_CACHE_DB_PATH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    claude_timeout_sec: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    progress_cost: bool = False,
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
) -> JudgeSummary:
    """Drive the judge over every document listed in a candidates manifest."""
    run_t0 = time.perf_counter()
    resolved_provider = normalize_llm_provider(llm_provider)
    resolved_model = resolve_model_for_provider(resolved_provider, model)
    resolved_input_mode = resolve_input_mode(resolved_provider, input_mode)
    dataset_root = resolve_dataset_root(target_dir)
    queries = load_queries(dataset_root / "queries.json", query_indices)
    judged_dir = dataset_root / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    manifest_docs = _load_manifest(Path(manifest_path), dataset_root=dataset_root)
    run_logger = _make_run_logger(
        log_dir=log_dir,
        dataset_root=dataset_root,
        llm_provider=resolved_provider,
        model=resolved_model,
        input_mode=resolved_input_mode,
        generation_mode=judge_mode,
        tag="gt_judge",
    )
    text_caller = (
        AzureResponsesTextCacheCaller(cache_db)
        if resolved_provider == "azure" and resolved_input_mode == "text"
        else None
    )
    claude_caller = (
        ClaudeCodeCacheCaller(cache_db) if resolved_provider == "claude-code" else None
    )

    results: list[JudgeDocResult] = []
    for manifest_doc in manifest_docs:
        doc_id = manifest_doc.doc_id
        output_path = judged_dir / f"{doc_id}.judged_answers.json"
        document_text = (
            _extract_pdf_text_for_prompt(manifest_doc.pdf_path)
            if resolved_input_mode == "text"
            else None
        )

        pending: list[QuerySpec] = []
        for query in queries:
            gt_key = str(query.idx)
            if output_path.exists() and _has_existing_judged(output_path, gt_key):
                _record_judge_result(
                    results,
                    JudgeDocResult(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="skipped_existing",
                        query_idx=query.idx,
                    ),
                    progress_cost=progress_cost,
                )
                continue
            if query.idx not in manifest_doc.candidates:
                _record_judge_result(
                    results,
                    JudgeDocResult(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="skipped_no_candidates",
                        query_idx=query.idx,
                    ),
                    progress_cost=progress_cost,
                )
                continue
            pending.append(query)

        if not pending:
            continue

        candidates_by_query = {q.idx: manifest_doc.candidates[q.idx] for q in pending}
        api_call_id = _judge_api_call_id(
            pdf_path=manifest_doc.pdf_path, queries=pending, judge_mode=judge_mode
        )

        try:
            judged_answers, cache_results = judge_document(
                pdf_path=manifest_doc.pdf_path,
                queries=pending,
                candidates_by_query=candidates_by_query,
                judge_mode=judge_mode,
                llm_provider=resolved_provider,
                model=resolved_model,
                input_mode=resolved_input_mode,
                cache_db=cache_db,
                max_tokens=max_tokens,
                claude_timeout_sec=claude_timeout_sec,
                text_caller=text_caller,
                claude_caller=claude_caller,
                document_text=document_text,
            )
        except Exception as exc:  # keep the batch moving across bad docs
            for query in pending:
                _record_judge_result(
                    results,
                    _doc_judge_result(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="failed",
                        query_idx=query.idx,
                        api_call_id=api_call_id,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    progress_cost=progress_cost,
                    run_logger=run_logger,
                )
            continue

        _merge_judged_answers(output_path, judged_answers)

        if judge_mode == "batched":
            cache_result = cache_results[0]
            for query, judged in zip(pending, judged_answers):
                _record_judge_result(
                    results,
                    _doc_judge_result(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="generated",
                        query_idx=query.idx,
                        response=cache_result,
                        judged=judged,
                        api_call_id=api_call_id,
                    ),
                    progress_cost=progress_cost,
                    run_logger=run_logger,
                )
        else:
            for query, judged, cache_result in zip(
                pending, judged_answers, cache_results
            ):
                _record_judge_result(
                    results,
                    _doc_judge_result(
                        doc_id=doc_id,
                        output_path=output_path,
                        status="generated",
                        query_idx=query.idx,
                        response=cache_result,
                        judged=judged,
                        api_call_id=f"per-query:{doc_id}:{query.idx}",
                    ),
                    progress_cost=progress_cost,
                    run_logger=run_logger,
                )

    summary = JudgeSummary(
        dataset_root=dataset_root,
        queries=tuple(queries),
        selected_count=len(manifest_docs),
        judged_count=sum(r.status == "generated" for r in results),
        skipped_existing_count=sum(r.status == "skipped_existing" for r in results),
        skipped_no_candidates_count=sum(
            r.status == "skipped_no_candidates" for r in results
        ),
        failed_count=sum(r.status == "failed" for r in results),
        run_latency_ms=_elapsed_ms(run_t0),
        log_path=run_logger.log_path if run_logger is not None else None,
        results=tuple(results),
    )
    if run_logger is not None:
        run_logger.log_run_summary(
            tuple(_judge_result_to_doc_run_result(r) for r in results),
            summary.run_latency_ms,
        )
        run_logger.close()
    return summary


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_judge_prompt(
    *,
    query: QuerySpec,
    candidates: Sequence[Candidate],
    doc_id: str,
    prompt_template: str | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="JUDGE",
        doc_id=doc_id,
        query_idx=query.idx,
        query_text=query.text,
        answer_type=query.answer_type,
        candidates_block=_format_candidates_block(candidates),
    )


def build_judge_text_prompt(
    *,
    query: QuerySpec,
    candidates: Sequence[Candidate],
    doc_id: str,
    document_text: str,
    prompt_template: str | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="JUDGE_TEXT",
        doc_id=doc_id,
        document_text=document_text,
        query_idx=query.idx,
        query_text=query.text,
        answer_type=query.answer_type,
        candidates_block=_format_candidates_block(candidates),
    )


def build_judge_all_prompt(
    *,
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
    doc_id: str,
    prompt_template: str | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="JUDGE_ALL",
        doc_id=doc_id,
        candidates_blocks=_format_candidates_blocks(queries, candidates_by_query),
    )


def build_judge_all_text_prompt(
    *,
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
    doc_id: str,
    document_text: str,
    prompt_template: str | None = None,
) -> str:
    return _render_gt_prompt(
        prompt_template=prompt_template,
        section="JUDGE_ALL_TEXT",
        doc_id=doc_id,
        document_text=document_text,
        candidates_blocks=_format_candidates_blocks(queries, candidates_by_query),
    )


def _format_candidates_block(candidates: Sequence[Candidate]) -> str:
    if not candidates:
        return "(no candidates provided)"
    parts: list[str] = []
    for i, c in enumerate(candidates):
        lines = [f"Candidate {i} (source={c.source}):"]
        lines.append(f"  answer: {json.dumps(c.answer, ensure_ascii=False)}")
        if c.reasoning:
            lines.append(f"  reasoning: {c.reasoning}")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def _format_candidates_blocks(
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
) -> str:
    chunks: list[str] = []
    for q in queries:
        candidates = candidates_by_query.get(q.idx, [])
        chunk = (
            f"\nQuestion index: {q.idx}\n"
            f"Question: {q.text}\n"
            f"Answer type: {q.answer_type}\n"
            "Candidates:\n"
            f"{_format_candidates_block(candidates)}"
        )
        chunks.append(chunk)
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


def _judge_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "picked_index": {"type": "integer"},
            "picked_source": {"type": "string"},
            "answer": _answer_value_schema(),
        },
        "required": [
            "reasoning",
            "picked_index",
            "picked_source",
            "answer",
        ],
    }


def _judges_response_schema(queries: Sequence[QuerySpec]) -> dict[str, Any]:
    allowed_query_indices = [query.idx for query in queries]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": len(allowed_query_indices),
                "maxItems": len(allowed_query_indices),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query_idx": {
                            "type": "integer",
                            "enum": allowed_query_indices,
                        },
                        "reasoning": {"type": "string"},
                        "picked_index": {"type": "integer"},
                        "picked_source": {"type": "string"},
                        "answer": _answer_value_schema(),
                    },
                    "required": [
                        "query_idx",
                        "reasoning",
                        "picked_index",
                        "picked_source",
                        "answer",
                    ],
                },
            }
        },
        "required": ["judgments"],
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_judge_response(
    raw_response: str, candidates: Sequence[Candidate]
) -> tuple[int, str, Any, str]:
    """Parse a per-query judge response into a tuple of fields.

    Returns (picked_index, picked_source, answer, reasoning). If
    picked_index is out of range, the result is normalized to (-1, "judge", ...).
    If picked_source disagrees with the candidate at picked_index, the source
    label from the candidate wins (the index is authoritative).
    """
    data = _load_judge_json(raw_response)
    for field_name in ("picked_index", "picked_source", "answer"):
        if field_name not in data:
            raise ValueError(
                f"Judge response missing '{field_name}': {raw_response[:200]}"
            )
    picked_index, picked_source = _normalize_pick(
        picked_index=int(data["picked_index"]),
        picked_source=str(data["picked_source"]),
        candidates=candidates,
    )
    return (
        picked_index,
        picked_source,
        data["answer"],
        str(data.get("reasoning", "")),
    )


def parse_judges_response(
    raw_response: str,
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
) -> dict[int, tuple[int, str, Any, str]]:
    """Parse a batched judge response. Mirrors generator.parse_answers_response validation."""
    data = _load_judge_json(raw_response)
    judgments = data.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError(
            f"Judge batched response missing 'judgments' array: {raw_response[:200]}"
        )
    expected = {q.idx for q in queries}
    result: dict[int, tuple[int, str, Any, str]] = {}
    for j in judgments:
        if not isinstance(j, dict):
            raise ValueError(f"Judgment entry is not an object: {j!r}")
        for field_name in ("query_idx", "picked_index", "picked_source", "answer"):
            if field_name not in j:
                raise ValueError(
                    f"Judgment entry missing '{field_name}' field: {j!r}"
                )
        qidx = int(j["query_idx"])
        if qidx not in expected:
            raise ValueError(
                f"Unexpected answer query_idx={qidx} in judge response"
            )
        if qidx in result:
            raise ValueError(f"Duplicate answer for query_idx={qidx}")
        candidates = candidates_by_query.get(qidx, [])
        picked_index, picked_source = _normalize_pick(
            picked_index=int(j["picked_index"]),
            picked_source=str(j["picked_source"]),
            candidates=candidates,
        )
        result[qidx] = (
            picked_index,
            picked_source,
            j["answer"],
            str(j.get("reasoning", "")),
        )
    missing = expected - set(result.keys())
    if missing:
        raise ValueError(
            f"Missing answers for queries {sorted(missing)} in judge response"
        )
    return result


def _normalize_pick(
    *,
    picked_index: int,
    picked_source: str,
    candidates: Sequence[Candidate],
) -> tuple[int, str]:
    if picked_index < 0 or picked_index >= len(candidates):
        return -1, "judge"
    candidate_source = candidates[picked_index].source
    if picked_source != candidate_source:
        return picked_index, candidate_source
    return picked_index, picked_source


def _load_judge_json(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if not text:
        raise ValueError("Judge response is empty")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Judge response did not contain JSON: {text[:200]}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"Judge response did not decode to an object: {text[:200]}")
    return obj


# ---------------------------------------------------------------------------
# Caller dispatch
# ---------------------------------------------------------------------------


def _call_judge_for_query_for_doc(
    *,
    pdf_path: Path,
    query: QuerySpec,
    candidates: Sequence[Candidate],
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    document_text: str | None = None,
) -> CacheResult:
    prompt_template = _load_judge_prompt_template(pdf_path)
    response_schema = _judge_response_schema()
    if input_mode == "text":
        resolved_document_text = (
            document_text
            if document_text is not None
            else _extract_pdf_text_for_prompt(pdf_path)
        )
        text_prompt = build_judge_text_prompt(
            query=query,
            candidates=candidates,
            doc_id=pdf_path.stem,
            document_text=resolved_document_text,
            prompt_template=prompt_template,
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
        prompt = build_judge_prompt(
            query=query,
            candidates=candidates,
            doc_id=pdf_path.stem,
            prompt_template=prompt_template,
        )
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


def _call_judge_for_queries_for_doc(
    *,
    pdf_path: Path,
    queries: Sequence[QuerySpec],
    candidates_by_query: Mapping[int, Sequence[Candidate]],
    llm_provider: str,
    model: str,
    input_mode: ResolvedInputMode,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
    max_tokens: int,
    claude_timeout_sec: int,
    document_text: str | None = None,
) -> CacheResult:
    prompt_template = _load_judge_prompt_template(pdf_path)
    response_schema = _judges_response_schema(queries)
    if input_mode == "text":
        resolved_document_text = (
            document_text
            if document_text is not None
            else _extract_pdf_text_for_prompt(pdf_path)
        )
        text_prompt = build_judge_all_text_prompt(
            queries=queries,
            candidates_by_query=candidates_by_query,
            doc_id=pdf_path.stem,
            document_text=resolved_document_text,
            prompt_template=prompt_template,
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
        prompt = build_judge_all_prompt(
            queries=queries,
            candidates_by_query=candidates_by_query,
            doc_id=pdf_path.stem,
            prompt_template=prompt_template,
        )
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


def _ensure_callers(
    resolved_provider: str,
    resolved_input_mode: ResolvedInputMode,
    cache_db: str,
    text_caller: AzureResponsesTextCacheCaller | None,
    claude_caller: ClaudeCodeCacheCaller | None,
) -> tuple[AzureResponsesTextCacheCaller | None, ClaudeCodeCacheCaller | None]:
    if (
        resolved_provider == "azure"
        and resolved_input_mode == "text"
        and text_caller is None
    ):
        text_caller = AzureResponsesTextCacheCaller(cache_db)
    if resolved_provider == "claude-code" and claude_caller is None:
        claude_caller = ClaudeCodeCacheCaller(cache_db)
    return text_caller, claude_caller


def _load_judge_prompt_template(pdf_path: Path) -> str:
    dataset_root = _dataset_root_from_pdf_path(pdf_path)
    return _load_gt_prompt_template(_dataset_name_for_log(dataset_root))


def _dataset_root_from_pdf_path(pdf_path: Path) -> Path:
    return pdf_path.parent.parent


def _dataset_name_for_log(dataset_root: Path) -> str:
    return (
        dataset_root.parent.name
        if dataset_root.name == "latest"
        else dataset_root.name
    )


# ---------------------------------------------------------------------------
# Manifest, output, and logging plumbing
# ---------------------------------------------------------------------------


def _load_manifest(
    manifest_path: Path, *, dataset_root: Path
) -> list[ManifestDocument]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_docs = data.get("documents")
    if not isinstance(raw_docs, list) or not raw_docs:
        raise ValueError(
            f"Manifest at {manifest_path} has no 'documents' array entries"
        )
    docs: list[ManifestDocument] = []
    for entry in raw_docs:
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest document entry is not an object: {entry!r}")
        doc_id = str(entry["doc_id"])
        pdf_path = Path(entry["pdf_path"])
        if not pdf_path.is_absolute():
            # Resolve relative paths against the manifest first, then dataset root.
            candidate = (manifest_path.parent / pdf_path).resolve()
            if candidate.exists():
                pdf_path = candidate
            else:
                pdf_path = (dataset_root / pdf_path).resolve()
        candidates: dict[int, tuple[Candidate, ...]] = {}
        for qidx_str, cand_list in (entry.get("candidates") or {}).items():
            qidx = int(qidx_str)
            candidates[qidx] = tuple(
                Candidate(
                    source=str(c["source"]),
                    answer=c.get("answer"),
                    reasoning=str(c.get("reasoning", "") or ""),
                    support=str(c.get("support", "") or ""),
                )
                for c in cand_list
            )
        docs.append(
            ManifestDocument(doc_id=doc_id, pdf_path=pdf_path, candidates=candidates)
        )
    return docs


def _merge_judged_answers(
    output_path: Path, judged: Sequence[JudgedAnswer]
) -> None:
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}
    for j in judged:
        data[str(j.query_idx)] = {
            "answer": j.answer,
            "picked_index": j.picked_index,
            "picked_source": j.picked_source,
            "reasoning": j.reasoning,
        }
    _atomic_write_json(output_path, data)


def _has_existing_judged(output_path: Path, gt_key: str) -> bool:
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    entry = data.get(gt_key)
    if isinstance(entry, dict):
        return "answer" in entry
    return entry is not None and entry != "" and entry != []


def _judge_api_call_id(
    *, pdf_path: Path, queries: Sequence[QuerySpec], judge_mode: JudgeMode
) -> str:
    query_part = ",".join(str(q.idx) for q in queries)
    return f"judge:{judge_mode}:{pdf_path.stem}:{query_part}"


def _record_judge_result(
    results: list[JudgeDocResult],
    result: JudgeDocResult,
    *,
    progress_cost: bool,
    run_logger=None,
) -> None:
    results.append(result)
    if run_logger is not None:
        # Reuse the gen logger via a DocRunResult-shaped view.
        run_logger.log_result(_judge_result_to_doc_run_result(result))
    if progress_cost and result.status == "generated":
        _print_judge_progress_cost(result, results)


def _doc_judge_result(
    *,
    doc_id: str,
    output_path: Path,
    status: str,
    query_idx: int,
    response: CacheResult | None = None,
    judged: JudgedAnswer | None = None,
    api_call_id: str = "",
    error: str = "",
) -> JudgeDocResult:
    return JudgeDocResult(
        doc_id=doc_id,
        output_path=output_path,
        status=status,
        query_idx=query_idx,
        cache_hit=response.cache_hit if response is not None else False,
        answer=judged.answer if judged is not None else None,
        picked_index=judged.picked_index if judged is not None else 0,
        picked_source=judged.picked_source if judged is not None else "",
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


def _judge_result_to_doc_run_result(result: JudgeDocResult):
    """Project a JudgeDocResult into the DocRunResult shape the logger expects."""
    from gt_gen.generator import DocRunResult

    return DocRunResult(
        doc_id=result.doc_id,
        output_path=result.output_path,
        status=result.status,
        query_idx=result.query_idx,
        cache_hit=result.cache_hit,
        answer=result.answer,
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        api_call_id=result.api_call_id,
        error=result.error,
    )


def _print_judge_progress_cost(
    result: JudgeDocResult, results: Sequence[JudgeDocResult]
) -> None:
    api_results = [r for r in results if r.status == "generated"]
    api_latency_ms = sum(r.latency_ms for r in api_results)
    api_cost_usd = sum(r.cost_usd for r in api_results)
    print(
        f"[judge] doc={result.doc_id} q={result.query_idx} "
        f"picked_idx={result.picked_index} src={result.picked_source} "
        f"lat={result.latency_ms:.1f}ms cost={_format_usd(result.cost_usd)} "
        f"total_lat={api_latency_ms:.1f}ms total_cost={_format_usd(api_cost_usd)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the universal LLM judge over candidate answers in a manifest "
            "and write picked answers to <dataset>/judged/<doc_id>.judged_answers.json."
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--candidates", required=True, help="Path to candidates manifest JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-idx", type=int)
    group.add_argument(
        "--query-indices",
        help="Comma-separated list, ranges like 1-5, or 'all'",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["per-query", "batched"],
        default=DEFAULT_JUDGE_MODE,
    )
    parser.add_argument(
        "--llm-provider",
        choices=["azure", "claude-code", "claude"],
        default=DEFAULT_LLM_PROVIDER,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--input-mode",
        choices=["auto", "text", "claude-read-pdf"],
        default=DEFAULT_INPUT_MODE,
    )
    parser.add_argument("--cache-db", default=DEFAULT_CACHE_DB_PATH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--claude-timeout-sec", type=int, default=DEFAULT_CLAUDE_TIMEOUT_SEC
    )
    parser.add_argument("--progress-cost", action="store_true")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    query_indices: list[int] | None
    if args.query_idx is not None:
        query_indices = [int(args.query_idx)]
    else:
        query_indices = _parse_query_indices_arg(args.query_indices)
    summary = judge_ground_truth_from_manifest(
        target_dir=args.target_dir,
        manifest_path=args.candidates,
        query_indices=query_indices,
        judge_mode=args.judge_mode,
        llm_provider=args.llm_provider,
        model=args.model,
        input_mode=args.input_mode,
        cache_db=args.cache_db,
        max_tokens=args.max_tokens,
        claude_timeout_sec=args.claude_timeout_sec,
        progress_cost=args.progress_cost,
        log_dir=args.log_dir,
    )
    _print_judge_summary(summary)
    return 0 if summary.failed_count == 0 else 1


def _print_judge_summary(summary: JudgeSummary) -> None:
    api_results = [r for r in summary.results if r.status == "generated"]
    api_latency_ms = sum(r.latency_ms for r in api_results)
    api_cost_usd = sum(r.cost_usd for r in api_results)
    print(
        "----- Judge Run Summary -----\n"
        f"Documents:    {summary.selected_count}\n"
        f"Judged:       {summary.judged_count}\n"
        f"Skipped:      {summary.skipped_existing_count} existing + "
        f"{summary.skipped_no_candidates_count} no-candidates\n"
        f"Failed:       {summary.failed_count}\n"
        f"Run Latency:  {summary.run_latency_ms:.3f} ms\n"
        f"API Latency:  {api_latency_ms:.3f} ms\n"
        f"API Cost:     {_format_usd(api_cost_usd)}\n"
        f"Log File:     {summary.log_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
