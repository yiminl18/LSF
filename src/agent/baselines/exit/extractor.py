"""ExitExtractor — wraps upstream ExitRAG.compress_documents + our API reader.

Paper: EXIT: Context-Aware Extractive Compression for RAG (ACL 2025 Findings)
Upstream: github.com/ThisIsHwang/EXIT  (vendored at upstream/EXIT/)

Pipeline:
  1. Text: doc_inputs.normalized_text (already populated by loader; loader
     itself falls back to PyMuPDF when the dataset has no reconstructed.json).
  2. Compress: upstream ExitRAG.compress_documents() via Gemma-2B PEFT.
     The Gemma adapter (doubleyyh/exit-gemma-2b) and base model
     (google/gemma-2b-it) must be available in the local Hugging Face cache;
     we do not provide a substitute classifier.
  3. Read: CachedLLMCaller with exit_answer_reader.txt prompt.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from agent.baselines.base import DocInputs, ExtractionResult
from agent.baselines.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost

_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "EXIT"
_READER_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_answer_reader.txt"
)

_GEMMA_HF_REPO = "doubleyyh/exit-gemma-2b"
_GEMMA_BASE_MODEL = "google/gemma-2b-it"
_DEFAULT_THRESHOLD = 0.5
_MAX_CONTEXT_CHARS = 4000
_MAX_ANSWER_TOKENS = 500

_COMPRESSOR_CACHE: dict[str, Any] = {}


def _get_compressor() -> Any:
    """Lazy-load upstream ExitRAG (compression components only — no reader model)."""
    if "rag" not in _COMPRESSOR_CACHE:
        if not _UPSTREAM_DIR.exists():
            raise RuntimeError(
                f"EXIT upstream not found at {_UPSTREAM_DIR}. "
                "Run: git submodule update --init "
                "src/agent/baselines/exit/upstream"
            )
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
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        llm_model: str = DEFAULT_LLM_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        total_cost = 0.0

        text = doc_inputs.normalized_text

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
                "classifier_method": "gemma_checkpoint",
                "context_chars": len(compressed_text),
                "threshold": threshold,
            },
            cost_usd=total_cost,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            gen_calls=1,  # reader only — Gemma classification is local
        )
