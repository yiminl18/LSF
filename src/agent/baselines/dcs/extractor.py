"""DCSExtractor — Dynamic Chunking and Selection baseline.

Pipeline:
  1. Chunk normalized text via semantic-similarity boundaries (chunker.py).
  2. Select top-k chunks relevant to the query (selector.py; MLP or LLM fallback).
  3. Read the concatenated chunks + query with dcs_answer_reader.txt prompt.

All LLM calls go through CachedLLMCaller to avoid re-billing cached pairs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.baselines.base import BaselineExtractor, DocInputs, ExtractionResult
from agent.baselines.dcs import chunker as _chunker
from agent.baselines.dcs import selector as _selector
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

_READER_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "dcs_answer_reader.txt"
)
_DEFAULT_TOP_K = 5
_MAX_ANSWER_TOKENS = 500


def _load_reader_prompt() -> str:
    return _READER_PROMPT_PATH.read_text(encoding="utf-8")


class DCSExtractor:
    name: str = "dcs"

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
        top_k: int = _DEFAULT_TOP_K,
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0

        # Step 1: chunk
        chunks = _chunker.chunk(
            text=doc_inputs.normalized_text,
            doc_id=doc_id,
        )

        # Step 2: select
        selected_chunks, selector_method = _selector.select_chunks(
            query=query_text,
            chunks=chunks,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k=top_k,
        )

        # Estimate cost of selector LLM call (zero for MLP; approximate for LLM)
        # The CachedLLMCaller already tracked it internally; we track cost separately
        # from reader call only.

        context = _selector.concat_chunks(selected_chunks)

        # Step 3: reader LLM call
        template = _load_reader_prompt()
        prompt = template.replace("{query}", query_text).replace("{context}", context)

        result = cached_caller.call(
            prompt,
            llm_provider=llm_provider,
            max_tokens=_MAX_ANSWER_TOKENS,
            model=llm_model,
        )
        reader_cost = compute_cost(
            result.input_tokens,
            result.output_tokens,
            llm_provider,
            model=llm_model,
        )
        total_cost += reader_cost

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ExtractionResult(
            generated_answer=result.response.strip(),
            trace={
                "n_chunks": len(chunks),
                "n_selected": len(selected_chunks),
                "selector_method": selector_method,
                "selected_chunk_texts": [c.text[:200] for c in selected_chunks],
                "context_chars": len(context),
            },
            cost_usd=total_cost,
            latency_ms=latency_ms,
        )
