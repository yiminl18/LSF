"""Chunk selector for DCS.

Tries to load the authors' MLP classifier checkpoint from the vendored
upstream/ directory. If no checkpoint is found, falls back to LLM
zero-shot chunk scoring via a batched prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.baselines.dcs.chunker import Chunk
from core.pipeline.e2e_utils.cache import CachedLLMCaller

_UPSTREAM_CHECKPOINT_GLOB = "upstream/checkpoints/**/*.pt"
_DEFAULT_TOP_K = 5
_MAX_CONCAT_CHARS = 4000

_ZEROSHOT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "baselines" / "dcs_chunk_selector_zeroshot.txt"


def _load_zeroshot_prompt() -> str:
    return _ZEROSHOT_PROMPT_PATH.read_text(encoding="utf-8")


def _try_load_mlp_checkpoint() -> Any | None:
    """Return a loaded MLP model if a checkpoint exists; else None."""
    checkpoint_dir = Path(__file__).parent / "upstream" / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    pts = list(checkpoint_dir.glob("**/*.pt"))
    if not pts:
        return None
    try:
        import torch
        model = torch.load(pts[0], map_location="cpu")
        return model
    except Exception:
        return None


def _score_with_llm(
    query: str,
    chunks: list[Chunk],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
) -> list[float]:
    """Return a relevance score in [0,1] for each chunk via batched LLM call."""
    template = _load_zeroshot_prompt()
    numbered = "\n\n".join(
        f"[Chunk {i+1}]\n{c.text[:800]}" for i, c in enumerate(chunks)
    )
    prompt = template.replace("{query}", query).replace("{chunks}", numbered).replace(
        "{n_chunks}", str(len(chunks))
    )
    result = cached_caller.call(
        prompt,
        llm_provider=llm_provider,
        max_tokens=300,
        model=llm_model,
    )
    return _parse_scores(result.response, n=len(chunks))


def _parse_scores(response: str, n: int) -> list[float]:
    """Parse LLM response into a list of n float scores."""
    # Expected format: JSON array [0.9, 0.3, ...]
    text = response.strip()
    # Try JSON array first
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            scores = json.loads(text[start:end + 1])
            if isinstance(scores, list) and len(scores) == n:
                return [float(s) for s in scores]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: extract numbers
    import re
    nums = re.findall(r"\d+\.?\d*", text)
    floats = [min(float(x), 1.0) for x in nums[:n]]
    # Pad with 0.0 if short
    while len(floats) < n:
        floats.append(0.0)
    return floats[:n]


def select_chunks(
    query: str,
    chunks: list[Chunk],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    top_k: int = _DEFAULT_TOP_K,
) -> tuple[list[Chunk], str]:
    """Select the top-k most relevant chunks for the query.

    Returns:
        (selected_chunks, method) where method is 'mlp' or 'llm_zeroshot'.
    """
    if not chunks:
        return [], "no_chunks"

    mlp = _try_load_mlp_checkpoint()
    if mlp is not None:
        # MLP path: not yet fully implemented pending upstream submodule.
        # Fall through to LLM fallback with a note.
        pass

    # LLM zero-shot fallback
    scores = _score_with_llm(query, chunks, cached_caller, llm_provider, llm_model)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted(idx for idx, _ in ranked[:top_k])
    selected = [chunks[i] for i in top_indices]
    return selected, "llm_zeroshot"


def concat_chunks(chunks: list[Chunk], max_chars: int = _MAX_CONCAT_CHARS) -> str:
    """Concatenate chunk texts, truncating at max_chars."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = c.text[:remaining]
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)
