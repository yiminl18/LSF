"""Sentence relevance classifier for EXIT.

Strategy (in priority order):
  1. Attempt to load the upstream Gemma-2B PEFT checkpoint from
     upstream/EXIT/checkpoints/ (follows EXIT paper's doubleyyh/exit-gemma-2b).
     Requires PyTorch + transformers + peft + GPU — will almost always be absent
     in test/CI environments.
  2. LLM zero-shot fallback via CachedLLMCaller using exit_sentence_relevance.txt.
     This preserves EXIT's "context-aware" property by passing neighbor sentences
     as context in each batch prompt.

The caller decides the threshold; default 0.5 maps "Yes" -> 1.0, "No" -> 0.0.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from core.pipeline.e2e_utils.cache import CachedLLMCaller

# Primary: local checkout under upstream/EXIT/checkpoints/
_CHECKPOINT_DIR = Path(__file__).parent / "upstream" / "EXIT" / "checkpoints"

# Secondary: HuggingFace hub cache (set EXIT_CHECKPOINT_DIR env var to override both)
_HF_CACHE_MODEL_ID = "doubleyyh/exit-gemma-2b"
_HF_CACHE_DIR = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub"
    / ("models--" + _HF_CACHE_MODEL_ID.replace("/", "--"))
)

_RELEVANCE_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "exit_sentence_relevance.txt"
)
_MAX_CLASSIFY_TOKENS = 10


def _load_relevance_prompt() -> str:
    return _RELEVANCE_PROMPT_PATH.read_text(encoding="utf-8")


def _find_gemma_checkpoint() -> Path | None:
    """Return the PEFT adapter directory, or None if not found.

    Search order:
    1. EXIT_CHECKPOINT_DIR env var (explicit override).
    2. upstream/EXIT/checkpoints/ (local copy in repo).
    3. HuggingFace hub cache at ~/.cache/huggingface/hub/models--doubleyyh--exit-gemma-2b/.
    """
    # 1. Explicit env-var override
    env_dir = os.environ.get("EXIT_CHECKPOINT_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "adapter_config.json").exists():
            return p
        # Accept a parent dir containing adapter_config.json
        hits = list(p.rglob("adapter_config.json"))
        if hits:
            return hits[0].parent

    # 2. Local checkout
    if _CHECKPOINT_DIR.exists():
        hits = list(_CHECKPOINT_DIR.rglob("adapter_config.json"))
        if hits:
            return hits[0].parent

    # 3. HuggingFace hub cache (snapshots/<sha>/)
    if _HF_CACHE_DIR.exists():
        hits = list(_HF_CACHE_DIR.rglob("adapter_config.json"))
        if hits:
            return hits[0].parent

    return None


def _gemma_checkpoint_available() -> bool:
    """Return True only when a Gemma PEFT checkpoint directory is present."""
    return _find_gemma_checkpoint() is not None


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
        return _classify_with_gemma(query, sentences, threshold)

    return _classify_with_llm(
        query=query,
        sentences=sentences,
        cached_caller=cached_caller,
        llm_provider=llm_provider,
        llm_model=llm_model,
        batch_size=batch_size,
    )


def _classify_with_gemma(
    query: str,
    sentences: list[str],
    threshold: float,
) -> list[float]:
    """Load PEFT Gemma checkpoint and score sentences locally."""
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import PeftModel  # type: ignore[import]

    # Find checkpoint via the unified search (local → HF cache → env override)
    checkpoint_path = _find_gemma_checkpoint()
    if checkpoint_path is None:
        raise RuntimeError(
            "Gemma PEFT checkpoint not found. "
            "Set EXIT_CHECKPOINT_DIR to the adapter directory, or ensure "
            f"the checkpoint is in {_CHECKPOINT_DIR} or {_HF_CACHE_DIR}."
        )
    base_model_id = "google/gemma-2b-it"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

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
            outputs = model.generate(
                **inputs, max_new_tokens=3, do_sample=False, temperature=1.0
            )
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(1.0 if decoded.strip().lower().startswith("yes") else 0.0)

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
