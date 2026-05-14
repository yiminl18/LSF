"""ExitExtractor — wraps upstream ExitRAG.compress_documents + our API reader.

Paper: EXIT: Context-Aware Extractive Compression for RAG (ACL 2025 Findings)
Upstream: github.com/ThisIsHwang/EXIT  (vendored at upstream/EXIT/)

Pipeline:
  1. Text: doc_inputs.normalized_text (PyMuPDF fallback if empty).
  2. Compress: upstream ExitRAG.compress_documents() via Gemma-2B PEFT, or
     LLM zero-shot fallback when checkpoint not in HF cache.
  3. Read: CachedLLMCaller with exit_answer_reader.txt prompt.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

from agent.baselines.base import DocInputs, ExtractionResult
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "EXIT"
_READER_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_answer_reader.txt"
)
_RELEVANCE_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_sentence_relevance.txt"
)

_GEMMA_HF_REPO = "doubleyyh/exit-gemma-2b"
_GEMMA_BASE_MODEL = "google/gemma-2b-it"
_DEFAULT_THRESHOLD = 0.5
_MAX_CONTEXT_CHARS = 4000
_MAX_ANSWER_TOKENS = 500
_LLM_CLASSIFY_TOKENS_PER_SENTENCE = 10
_LLM_BATCH_SIZE = 30

# Module-level cache — compressor loaded at most once per process
_COMPRESSOR_CACHE: dict[str, Any] = {}


def _gemma_checkpoint_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo_id=_GEMMA_HF_REPO, filename="adapter_config.json")
        return isinstance(result, str)
    except Exception:
        return False


def _get_compressor() -> Any:
    """Lazy-load upstream ExitRAG (compression components only — no reader model)."""
    if "rag" not in _COMPRESSOR_CACHE:
        if str(_UPSTREAM_DIR) not in sys.path:
            sys.path.insert(0, str(_UPSTREAM_DIR))
        from exit_rag import ExitRAG  # type: ignore[import]
        import torch
        import spacy
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        class _CompressorOnly(ExitRAG):
            """ExitRAG subclass — skips loading the Llama reader model."""
            def __init__(self) -> None:
                base = AutoModelForCausalLM.from_pretrained(
                    _GEMMA_BASE_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                self.exit_model = PeftModel.from_pretrained(base, _GEMMA_HF_REPO)
                self.exit_tokenizer = AutoTokenizer.from_pretrained(_GEMMA_BASE_MODEL)
                nlp = spacy.load(
                    "en_core_web_sm",
                    disable=["tok2vec", "tagger", "parser",
                              "attribute_ruler", "lemmatizer", "ner"],
                )
                nlp.enable_pipe("senter")
                self.nlp = nlp
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _COMPRESSOR_CACHE["rag"] = _CompressorOnly()

    return _COMPRESSOR_CACHE["rag"]


def _split_sentences(text: str) -> list[str]:
    """spaCy senter with regex fallback — mirrors upstream ExitRAG."""
    if not text or not text.strip():
        return []
    try:
        import spacy
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser",
                      "attribute_ruler", "lemmatizer", "ner"],
        )
        nlp.enable_pipe("senter")
        result = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        if result:
            return result
    except Exception:
        pass
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


def _llm_compress(
    query: str,
    text: str,
    threshold: float,
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> tuple[str, int, int, float]:
    """LLM zero-shot classification fallback.

    Returns (compressed_text, n_total, n_selected, cost_usd).
    """
    template = _RELEVANCE_PROMPT_PATH.read_text(encoding="utf-8")
    sentences = _split_sentences(text)
    if not sentences:
        return text[:_MAX_CONTEXT_CHARS], 0, 0, 0.0

    scores: list[float] = []
    total_cost = 0.0
    n = len(sentences)

    for start in range(0, n, _LLM_BATCH_SIZE):
        batch = sentences[start: start + _LLM_BATCH_SIZE]
        ctx_start = max(0, start - 1)
        ctx_end = min(n, start + _LLM_BATCH_SIZE + 1)
        context_window = " ".join(sentences[ctx_start:ctx_end])
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))
        prompt = (
            template
            .replace("{query}", query)
            .replace("{context}", context_window)
            .replace("{sentences}", numbered)
            .replace("{n}", str(len(batch)))
        )
        result = cached_caller.call(
            prompt,
            llm_provider=llm_provider,
            max_tokens=_LLM_CLASSIFY_TOKENS_PER_SENTENCE * len(batch),
            model=llm_model,
        )
        total_cost += compute_cost(
            result.input_tokens, result.output_tokens, llm_provider, model=llm_model,
        )
        tokens = re.findall(r"\b(yes|no)\b", result.response.lower())
        batch_scores = [1.0 if t == "yes" else 0.0 for t in tokens[: len(batch)]]
        while len(batch_scores) < len(batch):
            batch_scores.append(0.0)
        scores.extend(batch_scores)

    selected: list[str] = []
    total_chars = 0
    for sent, score in zip(sentences, scores):
        if score > threshold:
            if total_chars + len(sent) > _MAX_CONTEXT_CHARS:
                break
            selected.append(sent)
            total_chars += len(sent) + 1

    if not selected:
        selected = [" ".join(sentences)[:_MAX_CONTEXT_CHARS]]

    n_selected = sum(1 for s in scores if s > threshold)
    return " ".join(selected), n, n_selected, total_cost


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
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0

        text = doc_inputs.normalized_text
        if not text and doc_inputs.pdf_path.exists():
            import fitz
            text = "\n\n".join(page.get_text() for page in fitz.open(str(doc_inputs.pdf_path)))

        if _gemma_checkpoint_available() and _UPSTREAM_DIR.exists():
            if str(_UPSTREAM_DIR) not in sys.path:
                sys.path.insert(0, str(_UPSTREAM_DIR))
            from exit_rag import Document as ExitDocument  # type: ignore[import]
            rag = _get_compressor()
            compressed_text, selections, _scores = rag.compress_documents(
                query_text,
                [ExitDocument(title=doc_id, text=text)],
                threshold=threshold,
            )
            if not compressed_text:
                compressed_text = text[:_MAX_CONTEXT_CHARS]
            n_total = len(selections)
            n_selected = sum(1 for s in selections if s)
            classifier_method = "gemma_checkpoint"
        else:
            compressed_text, n_total, n_selected, classify_cost = _llm_compress(
                query=query_text,
                text=text,
                threshold=threshold,
                cached_caller=cached_caller,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            total_cost += classify_cost
            classifier_method = "llm_zeroshot"

        template = _READER_PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.replace("{query}", query_text).replace("{context}", compressed_text)
        reader_result = cached_caller.call(
            prompt,
            llm_provider=llm_provider,
            max_tokens=_MAX_ANSWER_TOKENS,
            model=llm_model,
        )
        total_cost += compute_cost(
            reader_result.input_tokens, reader_result.output_tokens,
            llm_provider, model=llm_model,
        )

        return ExtractionResult(
            generated_answer=reader_result.response.strip(),
            trace={
                "n_sentences_total": n_total,
                "n_sentences_selected": n_selected,
                "classifier_method": classifier_method,
                "context_chars": len(compressed_text),
                "threshold": threshold,
            },
            cost_usd=total_cost,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
