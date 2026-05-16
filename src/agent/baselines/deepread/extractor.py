"""DeepReadExtractor — locate-then-read baseline (arXiv:2602.05014).

Pipeline:
  1. LLMOCR.parse_pdf(pdf_path) -> ParagraphIndex (cached on disk).
  2. ReAct loop (<=6 turns): system prompt with Retrieve + ReadSection tools.
  3. Final answer call assembles evidence and emits a span.

All LLM calls go through CachedLLMCaller.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs, ExtractionResult
from agent.baselines.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from agent.baselines.deepread.index import ParagraphIndex
from agent.baselines.deepread.ocr import DEFAULT_OCR_MODEL, DEFAULT_OCR_PROVIDER, LLMOCR
from agent.baselines.deepread.tools import retrieve, read_section
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

_LOCATE_READ_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "deepread_locate_read.txt"
)
_MAX_TURNS = 6
_MAX_ANSWER_TOKENS = 500
_MAX_TOOL_TOKENS = 800


def _load_locate_read_prompt() -> str:
    return _LOCATE_READ_PROMPT_PATH.read_text(encoding="utf-8")


def _format_hits(hits: list) -> str:
    if not hits:
        return "No results found."
    lines = []
    for h in hits:
        lines.append(
            f"[section_id={h.coord.section_id}, order={h.coord.in_section_order}] "
            f"score={h.score:.3f}\n{h.snippet}"
        )
    return "\n\n".join(lines)


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract the first tool call from model output.

    Expects JSON like: {"tool": "Retrieve", "args": {"query": "...", "k": 5}}
    or               : {"tool": "ReadSection", "args": {"section_id": 1, "start": 0}}
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        if isinstance(obj, dict) and "tool" in obj:
            return obj.get("tool", ""), obj.get("args", {})
    except (json.JSONDecodeError, ValueError):
        pass
    return None


class DeepReadExtractor:
    name: str = "deepread"

    def __init__(
        self,
        ocr_model: str = DEFAULT_OCR_MODEL,
        ocr_provider: str = DEFAULT_OCR_PROVIDER,
        max_pages: int | None = None,
    ) -> None:
        self._ocr_model = ocr_model
        self._ocr_provider = ocr_provider
        self._max_pages = max_pages

    def extract(
        self,
        *,
        query_idx: int,
        query_text: str,
        doc_id: str,
        doc_inputs: DocInputs,
        cached_caller: CachedLLMCaller,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0
        gen_calls = 0

        # Step 1: OCR — the vision call doesn't share the SQLite cache, so we
        # add OCR cost back into total_cost via LLMOCR.last_cost_usd. Each OCR
        # page is one generation call when not cached; LLMOCR exposes this so
        # the runner's call-count summary stays accurate.
        ocr = LLMOCR(
            cached_caller,
            ocr_model=self._ocr_model,
            ocr_provider=self._ocr_provider,
            max_pages=self._max_pages,
        )
        index = ocr.parse_pdf(doc_inputs.pdf_path, doc_id=doc_id)
        total_cost += ocr.last_cost_usd
        gen_calls += getattr(ocr, "last_call_count", 0)

        if not index.paragraphs:
            # PDF could not be parsed (e.g., missing file); fall back to normalized text.
            index = _build_fallback_index(doc_inputs.normalized_text)

        # Step 2: ReAct loop
        # Seed evidence with a free Retrieve call so the model always has something
        # to reason over on turn 1 — small models skip tool use when evidence is empty.
        system_prompt = _load_locate_read_prompt()
        conversation: list[dict[str, str]] = []
        initial_hits = retrieve(index, query_text, k=5)
        evidence_parts: list[str] = [f"[Retrieve]\n{_format_hits(initial_hits)}"]
        tool_calls: list[dict[str, Any]] = [{"turn": -1, "tool": "Retrieve", "args": {"query": query_text, "k": 5}}]

        for turn in range(_MAX_TURNS):
            obs_str = "\n\n".join(evidence_parts) if evidence_parts else "No evidence collected yet."
            user_msg = (
                f"Question: {query_text}\n\n"
                f"Evidence so far:\n{obs_str}\n\n"
                "Issue a tool call (JSON) to retrieve more evidence, or output your final answer "
                'prefixed with "FINAL ANSWER:". Available tools:\n'
                '{"tool": "Retrieve", "args": {"query": "...", "k": 5}}\n'
                '{"tool": "ReadSection", "args": {"section_id": <int>, "start": 0, "end": 5}}\n'
            )
            prompt = system_prompt + "\n\n" + user_msg

            result = cached_caller.call(
                prompt,
                llm_provider=llm_provider,
                max_tokens=_MAX_TOOL_TOKENS,
                model=llm_model,
            )
            gen_calls += 1
            call_cost = compute_cost(
                result.input_tokens,
                result.output_tokens,
                llm_provider,
                model=llm_model,
            )
            total_cost += call_cost
            response = result.response.strip()

            if "FINAL ANSWER:" in response:
                final_answer = response.split("FINAL ANSWER:", 1)[1].strip()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return ExtractionResult(
                    generated_answer=final_answer,
                    trace={
                        "turns": turn + 1,
                        "tool_calls": tool_calls,
                        "evidence_parts": evidence_parts,
                    },
                    cost_usd=total_cost,
                    latency_ms=latency_ms,
                    gen_calls=gen_calls,
                )

            parsed = _parse_tool_call(response)
            if parsed is None:
                # Treat entire response as final answer if no tool call found
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return ExtractionResult(
                    generated_answer=response,
                    trace={
                        "turns": turn + 1,
                        "tool_calls": tool_calls,
                        "evidence_parts": evidence_parts,
                        "early_exit": "no_tool_call",
                    },
                    cost_usd=total_cost,
                    latency_ms=latency_ms,
                    gen_calls=gen_calls,
                )

            tool_name, args = parsed
            tool_calls.append({"turn": turn, "tool": tool_name, "args": args})

            if tool_name == "Retrieve":
                hits = retrieve(index, args.get("query", query_text), k=int(args.get("k", 5)))
                obs = _format_hits(hits)
            elif tool_name == "ReadSection":
                obs = read_section(
                    index,
                    section_id=int(args.get("section_id", 0)),
                    start=int(args.get("start", 0)),
                    end=args.get("end"),
                )
            else:
                obs = f"Unknown tool: {tool_name}"

            evidence_parts.append(f"[{tool_name}]\n{obs}")

        # Exhausted turns — do a final answer call with accumulated evidence.
        # Use the same "\n\n" join as the in-loop user message; the two-space
        # join here was a typo that flattened evidence into one wall of text.
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence collected."
        final_prompt = (
            system_prompt
            + "\n\n"
            + f"Question: {query_text}\n\n"
            + f"Evidence:\n{evidence_text}\n\n"
            + "FINAL ANSWER:"
        )
        final_result = cached_caller.call(
            final_prompt,
            llm_provider=llm_provider,
            max_tokens=_MAX_ANSWER_TOKENS,
            model=llm_model,
        )
        gen_calls += 1
        total_cost += compute_cost(
            final_result.input_tokens,
            final_result.output_tokens,
            llm_provider,
            model=llm_model,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ExtractionResult(
            generated_answer=final_result.response.strip(),
            trace={
                "turns": _MAX_TURNS,
                "tool_calls": tool_calls,
                "evidence_parts": evidence_parts,
                "early_exit": "max_turns",
            },
            cost_usd=total_cost,
            latency_ms=latency_ms,
            gen_calls=gen_calls,
        )


def _build_fallback_index(normalized_text: str) -> ParagraphIndex:
    """Build a minimal ParagraphIndex from plain normalized text when OCR fails."""
    from agent.baselines.deepread.index import (
        ParagraphIndex,
        Paragraph,
        ParagraphCoord,
        SectionMeta,
    )
    idx = ParagraphIndex()
    idx.sections[0] = SectionMeta(section_id=0, heading="Document", level=1, page_no=1)
    # Split into ~500-char paragraphs
    chunk_size = 500
    text = normalized_text.strip()
    for i in range(0, len(text), chunk_size):
        para = Paragraph(
            coord=ParagraphCoord(section_id=0, in_section_order=i // chunk_size),
            text=text[i:i + chunk_size],
            page_no=1,
        )
        idx.paragraphs.append(para)
    return idx
