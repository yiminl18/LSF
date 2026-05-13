"""ExitExtractor — extractive context compression for RAG (arXiv:2412.12559).

Paper: EXIT: Context-Aware Extractive Compression for Enhancing RAG
       ACL 2025 Findings, https://github.com/ThisIsHwang/EXIT

Pipeline:
  1. Split doc_inputs.normalized_text into sentences (sentences.py).
  2. Classify sentences for relevance using EXIT's context-aware binary classifier
     (classifier.py: Gemma PEFT checkpoint when present, LLM zero-shot fallback).
  3. Select sentences with score > threshold; cap at _MAX_CONTEXT_CHARS; preserve order.
  4. Feed selected context + query to a reader LLM (exit_answer_reader.txt).

Deviation from paper: EXIT's original pipeline classifies retrieved-and-ranked
candidate chunks from a retrieval system. Here, the whole document's normalized
text serves as the candidate pool (mirroring how DCS treated normalized_text),
since no separate retrieval stage exists in the LSF baseline pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

from agent.baselines.base import BaselineExtractor, DocInputs, ExtractionResult
from agent.baselines.exit import classifier as _classifier
from agent.baselines.exit import sentences as _sentences
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

_READER_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_answer_reader.txt"
)
_DEFAULT_THRESHOLD = 0.5
_MAX_CONTEXT_CHARS = 4000
_DEFAULT_BATCH_SIZE = 30
_MAX_ANSWER_TOKENS = 500


def _load_reader_prompt() -> str:
    return _READER_PROMPT_PATH.read_text(encoding="utf-8")


class ExitExtractor:
    name: str = "exit"

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
        threshold: float = _DEFAULT_THRESHOLD,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0

        # Step 1: sentence splitting
        all_sentences = _sentences.split_sentences(doc_inputs.normalized_text)

        # Step 2: classify each sentence for relevance (context-aware, in batches)
        scores, classifier_cost = _classifier.classify_sentences(
            query=query_text,
            sentences=all_sentences,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
            batch_size=batch_size,
            threshold=threshold,
        )
        total_cost += classifier_cost

        # Step 3: select relevant sentences, preserve document order, cap length
        selected: list[str] = []
        total_chars = 0
        for sentence, score in zip(all_sentences, scores):
            if score > threshold:
                if total_chars + len(sentence) > _MAX_CONTEXT_CHARS:
                    break
                selected.append(sentence)
                total_chars += len(sentence) + 1  # +1 for space separator

        # If nothing selected (e.g., all scores == 0), fall back to first N chars
        if not selected and all_sentences:
            fallback = " ".join(all_sentences)[:_MAX_CONTEXT_CHARS]
            selected = [fallback]

        context = " ".join(selected)

        # Step 4: reader LLM call
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

        classifier_method = (
            "gemma_checkpoint"
            if _classifier._gemma_checkpoint_available()
            else "llm_zeroshot"
        )

        return ExtractionResult(
            generated_answer=result.response.strip(),
            trace={
                "n_sentences_total": len(all_sentences),
                "n_sentences_selected": len(selected),
                "classifier_method": classifier_method,
                "context_chars": len(context),
                "threshold": threshold,
            },
            cost_usd=total_cost,
            latency_ms=latency_ms,
        )
