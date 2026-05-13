"""Sentence relevance classifier for EXIT.

Strategy (in priority order):
  1. Attempt to load the Gemma-2B PEFT checkpoint via HuggingFace hub
     (repo "doubleyyh/exit-gemma-2b"). Uses huggingface_hub.try_to_load_from_cache
     to detect a local cache hit without downloading. Requires PyTorch +
     transformers + peft — will be absent in test/CI environments unless the
     model has been explicitly downloaded.
  2. LLM zero-shot fallback via CachedLLMCaller using exit_sentence_relevance.txt.
     This preserves EXIT's "context-aware" property by passing neighbor sentences
     as context in each batch prompt.

The caller decides the threshold; default 0.5.  Gemma path returns softmax
probability of the "Yes" token; LLM path maps "yes" → 1.0, "no" → 0.0.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.pipeline.e2e_utils.cache import CachedLLMCaller

# HuggingFace repo ID for the EXIT PEFT adapter (base + adapter loaded separately)
_GEMMA_HF_REPO: str = "doubleyyh/exit-gemma-2b"
# Base model declared in adapter_config.json → base_model_name_or_path
_GEMMA_BASE_MODEL: str = "google/gemma-2b-it"

_RELEVANCE_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_sentence_relevance.txt"
)
_MAX_CLASSIFY_TOKENS = 10

# Class-scope cache so we don't reload the model on every classify_sentences call
_GEMMA_MODEL_CACHE: dict[str, Any] = {}


def _load_relevance_prompt() -> str:
    return _RELEVANCE_PROMPT_PATH.read_text(encoding="utf-8")


def _gemma_checkpoint_available() -> bool:
    """Return True when adapter_config.json is present in the local HF cache.

    Uses huggingface_hub.try_to_load_from_cache — returns a path string on
    cache hit, or a sentinel (LIBRARY_NOT_FOUND / None) on miss.  Never
    triggers a download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import]
        result = try_to_load_from_cache(
            repo_id=_GEMMA_HF_REPO,
            filename="adapter_config.json",
        )
        # Returns a str path on hit; returns None or _LIBRARY_NOT_FOUND sentinel on miss
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def classify_sentences(
    query: str,
    sentences: list[str],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    batch_size: int = 30,
    threshold: float = 0.5,
) -> list[float]:
    """Score each sentence in *sentences* for relevance to *query*.

    Returns a list of floats in [0.0, 1.0] parallel to *sentences*.
    Score > threshold means "keep".

    Uses the Gemma checkpoint if available, otherwise LLM zero-shot fallback.
    """
    if not sentences:
        return []

    if _gemma_checkpoint_available():
        return _classify_with_gemma(query, sentences)

    return _classify_with_llm(
        query=query,
        sentences=sentences,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        batch_size=batch_size,
    )


def _get_gemma_model_and_tokenizer() -> tuple[Any, Any]:
    """Lazy-load and cache the Gemma base model + PEFT adapter at class scope."""
    if "model" not in _GEMMA_MODEL_CACHE:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        from peft import PeftModel  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(_GEMMA_BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            _GEMMA_BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base, _GEMMA_HF_REPO)
        model.eval()

        _GEMMA_MODEL_CACHE["model"] = model
        _GEMMA_MODEL_CACHE["tokenizer"] = tokenizer

    return _GEMMA_MODEL_CACHE["model"], _GEMMA_MODEL_CACHE["tokenizer"]


def _classify_with_gemma(
    query: str,
    sentences: list[str],
) -> list[float]:
    """Load PEFT Gemma checkpoint from HF cache and score sentences locally.

    Prompt format matches upstream exit_rag.py exactly.  Uses logit-level
    softmax over ("Yes", "No") token IDs — returns probability of "Yes" as
    a float in [0, 1] per sentence.
    """
    import torch  # type: ignore[import]

    model, tokenizer = _get_gemma_model_and_tokenizer()

    # Encode Yes/No token IDs once
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]

    full_context = " ".join(sentences)
    scores: list[float] = []

    for sentence in sentences:
        prompt = (
            f"<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{full_context}\n"
            f"Sentence:\n{sentence}\n"
            f'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits at final position for Yes/No tokens → softmax probability
            logits = outputs.logits[0, -1, [yes_id, no_id]]
            prob = torch.softmax(logits, dim=0)[0].item()
        scores.append(prob)

    return scores


def _classify_with_llm(
    query: str,
    sentences: list[str],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    batch_size: int,
) -> list[float]:
    """LLM zero-shot fallback: classify sentences in batches preserving neighbor context."""
    template = _load_relevance_prompt()
    scores: list[float] = []

    n = len(sentences)
    for start in range(0, n, batch_size):
        batch = sentences[start : start + batch_size]
        # Neighbor context: include one sentence before and after the batch window
        ctx_start = max(0, start - 1)
        ctx_end = min(n, start + batch_size + 1)
        context_window = " ".join(sentences[ctx_start:ctx_end])

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
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
            max_tokens=_MAX_CLASSIFY_TOKENS * len(batch),
            model=llm_model,
        )
        batch_scores = _parse_scores(result.response, len(batch))
        scores.extend(batch_scores)

    return scores


def _parse_scores(response: str, expected_n: int) -> list[float]:
    """Parse LLM response into per-sentence scores.

    Accepts two formats:
      - Comma/space-separated "yes/no" tokens: "yes, no, yes"
      - Numbered list: "1. yes\n2. no\n3. yes"
    Falls back to 0.0 for any unparseable positions.
    """
    # Try to find yes/no tokens in order
    tokens = re.findall(r"\b(yes|no)\b", response.lower())
    if tokens:
        result = [1.0 if t == "yes" else 0.0 for t in tokens[:expected_n]]
        # Pad if fewer tokens than expected
        while len(result) < expected_n:
            result.append(0.0)
        return result

    # Fallback: return all zeros
    return [0.0] * expected_n
