"""DeepRead locate-then-read extractor adapted from chiyu-dev."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baseline.deepread.index import Paragraph, ParagraphCoord, ParagraphIndex, SectionMeta
from baseline.deepread.llm import DEFAULT_MODEL, DEFAULT_PROVIDER, chat_text, resolve_model
from baseline.deepread.ocr import DEFAULT_OCR_MODEL, DEFAULT_OCR_PROVIDER, LLMOCR
from baseline.deepread.tools import read_section, retrieve

_LOCATE_READ_PROMPT_PATH = Path(__file__).with_name("deepread_locate_read.txt")
_MAX_TURNS = 6
_MAX_TOOL_TOKENS = 800
_MAX_ANSWER_TOKENS = 500


@dataclass(slots=True)
class DeepReadResult:
    answer: str | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    total_cost_usd: float
    model: str
    gen_calls: int
    trace: dict[str, Any]


def _load_locate_read_prompt() -> str:
    return _LOCATE_READ_PROMPT_PATH.read_text(encoding="utf-8")


def _format_hits(hits: list) -> str:
    if not hits:
        return "No results found."
    lines: list[str] = []
    for h in hits:
        lines.append(
            f"[section_id={h.coord.section_id}, order={h.coord.in_section_order}] "
            f"score={h.score:.3f}\n{h.snippet}"
        )
    return "\n\n".join(lines)


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    args = obj.get("args", {})
    return str(obj.get("tool", "")), args if isinstance(args, dict) else {}


def _build_fallback_index(text: str) -> ParagraphIndex:
    idx = ParagraphIndex()
    idx.sections[0] = SectionMeta(section_id=0, heading="Document", level=1, page_no=1)
    chunk_size = 500
    text = text.strip()
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if not chunk:
            continue
        idx.paragraphs.append(
            Paragraph(
                coord=ParagraphCoord(section_id=0, in_section_order=i // chunk_size),
                text=chunk,
                page_no=1,
            )
        )
    return idx


class DeepReadExtractor:
    name = "deepread"

    def __init__(
        self,
        *,
        ocr_model: str = DEFAULT_OCR_MODEL,
        ocr_provider: str = DEFAULT_OCR_PROVIDER,
        max_pages: int | None = None,
    ) -> None:
        self._ocr_model = resolve_model(ocr_model)
        self._ocr_provider = ocr_provider
        self._max_pages = max_pages

    def extract(
        self,
        *,
        pdf_path: Path,
        question: str,
        doc_id: str,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        fallback_text: str = "",
    ) -> DeepReadResult:
        t0 = time.perf_counter()
        resolved_model = resolve_model(model)
        total_cost = 0.0
        input_tokens = 0
        output_tokens = 0
        gen_calls = 0

        ocr_error: str | None = None
        ocr = LLMOCR(
            ocr_model=self._ocr_model,
            ocr_provider=self._ocr_provider,
            max_pages=self._max_pages,
        )
        try:
            index = ocr.parse_pdf(pdf_path, doc_id=doc_id)
            total_cost += ocr.last_cost_usd
            input_tokens += ocr.last_input_tokens
            output_tokens += ocr.last_output_tokens
            gen_calls += ocr.last_call_count
        except Exception as exc:
            if not fallback_text.strip():
                raise
            ocr_error = str(exc)
            index = ParagraphIndex()

        if not index.paragraphs and fallback_text.strip():
            index = _build_fallback_index(fallback_text)

        if not index.paragraphs:
            return DeepReadResult(
                answer=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=round(time.perf_counter() - t0, 3),
                total_cost_usd=total_cost,
                model=resolved_model,
                gen_calls=gen_calls,
                trace={
                    "ocr_cache_hit": ocr.last_cache_hit,
                    "ocr_cache_path": str(ocr.last_cache_path) if ocr.last_cache_path else None,
                    "ocr_error": ocr_error,
                    "error": "no paragraphs produced",
                },
            )

        system_prompt = _load_locate_read_prompt()
        initial_hits = retrieve(index, question, k=5)
        evidence_parts = [f"[Retrieve]\n{_format_hits(initial_hits)}"]
        tool_calls: list[dict[str, Any]] = [
            {"turn": -1, "tool": "Retrieve", "args": {"query": question, "k": 5}}
        ]

        for turn in range(_MAX_TURNS):
            obs_str = "\n\n".join(evidence_parts) if evidence_parts else "No evidence collected yet."
            prompt = (
                system_prompt
                + "\n\n"
                + f"Question: {question}\n\n"
                + f"Evidence so far:\n{obs_str}\n\n"
                + "Issue a tool call (JSON) to retrieve more evidence, or output your final answer "
                + 'prefixed with "FINAL ANSWER:". Available tools:\n'
                + '{"tool": "Retrieve", "args": {"query": "...", "k": 5}}\n'
                + '{"tool": "ReadSection", "args": {"section_id": <int>, "start": 0, "end": 5}}\n'
            )
            result = chat_text(
                prompt,
                provider=provider,
                model=resolved_model,
                max_tokens=_MAX_TOOL_TOKENS,
            )
            total_cost += result.cost_usd
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            gen_calls += 1
            response = result.text.strip()

            if "FINAL ANSWER:" in response:
                answer = response.split("FINAL ANSWER:", 1)[1].strip()
                return DeepReadResult(
                    answer=answer,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=round(time.perf_counter() - t0, 3),
                    total_cost_usd=total_cost,
                    model=resolved_model,
                    gen_calls=gen_calls,
                    trace={
                        "turns": turn + 1,
                        "tool_calls": tool_calls,
                        "evidence_parts": evidence_parts,
                        "ocr_cache_hit": ocr.last_cache_hit,
                        "ocr_cache_path": str(ocr.last_cache_path) if ocr.last_cache_path else None,
                        "ocr_error": ocr_error,
                    },
                )

            parsed = _parse_tool_call(response)
            if parsed is None:
                return DeepReadResult(
                    answer=response or None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=round(time.perf_counter() - t0, 3),
                    total_cost_usd=total_cost,
                    model=resolved_model,
                    gen_calls=gen_calls,
                    trace={
                        "turns": turn + 1,
                        "tool_calls": tool_calls,
                        "evidence_parts": evidence_parts,
                        "early_exit": "no_tool_call",
                        "ocr_cache_hit": ocr.last_cache_hit,
                        "ocr_cache_path": str(ocr.last_cache_path) if ocr.last_cache_path else None,
                        "ocr_error": ocr_error,
                    },
                )

            tool_name, args = parsed
            tool_calls.append({"turn": turn, "tool": tool_name, "args": args})

            if tool_name == "Retrieve":
                hits = retrieve(index, str(args.get("query", question)), k=int(args.get("k", 5)))
                obs = _format_hits(hits)
            elif tool_name == "ReadSection":
                end_raw = args.get("end")
                obs = read_section(
                    index,
                    section_id=int(args.get("section_id", 0)),
                    start=int(args.get("start", 0)),
                    end=int(end_raw) if end_raw is not None else None,
                )
            else:
                obs = f"Unknown tool: {tool_name}"

            evidence_parts.append(f"[{tool_name}]\n{obs}")

        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence collected."
        final_prompt = (
            system_prompt
            + "\n\n"
            + f"Question: {question}\n\n"
            + f"Evidence:\n{evidence_text}\n\n"
            + "FINAL ANSWER:"
        )
        final_result = chat_text(
            final_prompt,
            provider=provider,
            model=resolved_model,
            max_tokens=_MAX_ANSWER_TOKENS,
        )
        total_cost += final_result.cost_usd
        input_tokens += final_result.input_tokens
        output_tokens += final_result.output_tokens
        gen_calls += 1
        return DeepReadResult(
            answer=final_result.text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=round(time.perf_counter() - t0, 3),
            total_cost_usd=total_cost,
            model=resolved_model,
            gen_calls=gen_calls,
            trace={
                "turns": _MAX_TURNS,
                "tool_calls": tool_calls,
                "evidence_parts": evidence_parts,
                "early_exit": "max_turns",
                "ocr_cache_hit": ocr.last_cache_hit,
                "ocr_cache_path": str(ocr.last_cache_path) if ocr.last_cache_path else None,
                "ocr_error": ocr_error,
            },
        )
