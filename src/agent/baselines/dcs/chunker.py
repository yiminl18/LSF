"""Semantic-similarity adaptive chunker for DCS.

Algorithm (per arXiv:2506.00773):
  1. Split normalized text into sentences.
  2. Embed each sentence with sentence-transformers paraphrase-multilingual-MiniLM-L12-v2.
  3. Compute cosine similarity between adjacent sentence pairs.
  4. Place chunk boundaries at local minima below `sim_threshold`.
  5. Merge adjacent short chunks until each chunk reaches ~`target_tokens` tokens.

Model is lazy-loaded so import-time does not trigger the 470 MB download.
Embeddings are cached on disk as .cache/dcs/<doc_id>.npz.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_CACHE_DIR = Path(".cache/dcs")
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Boundary placed when adjacent cosine similarity is below this value.
_DEFAULT_SIM_THRESHOLD = 0.5
# Target average chunk size in characters (~256 tokens * 4 chars/token).
_DEFAULT_TARGET_CHARS = 1024
_SENTENCE_SPLITTER_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Chunk:
    text: str
    sentence_indices: list[int]


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLITTER_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _cache_path(doc_id: str) -> Path:
    return _CACHE_DIR / f"{doc_id}.npz"


def _embed(sentences: list[str], doc_id: str, model: Any) -> "np.ndarray":
    """Embed sentences, using disk cache when available."""
    import numpy as np
    cache = _cache_path(doc_id)
    text_hash = hashlib.sha256("\n".join(sentences).encode()).hexdigest()

    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data.get("hash", b"").tolist()) == text_hash:
            return data["embeddings"]

    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, embeddings=embeddings, hash=np.bytes_(text_hash))
    return embeddings


def _load_model() -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def chunk(
    text: str,
    doc_id: str,
    sim_threshold: float = _DEFAULT_SIM_THRESHOLD,
    target_chars: int = _DEFAULT_TARGET_CHARS,
    model: Any = None,
) -> list[Chunk]:
    """Chunk `text` using semantic similarity boundaries.

    Args:
        text: Normalized document text.
        doc_id: Used for on-disk embedding cache key.
        sim_threshold: Boundary placed when adjacent sim < threshold.
        target_chars: Merge adjacent short chunks to reach this size.
        model: Optional pre-loaded SentenceTransformer (pass in tests to skip download).

    Returns:
        List of Chunk objects, ordered by position, non-overlapping.
    """
    import numpy as np

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], sentence_indices=[0])]

    if model is None:
        model = _load_model()

    embeddings = _embed(sentences, doc_id, model)

    # Compute adjacent similarities
    sims: list[float] = []
    for i in range(len(sentences) - 1):
        sims.append(_cosine_sim(embeddings[i], embeddings[i + 1]))

    # Place boundaries at local minima below threshold
    boundaries: set[int] = set()
    for i, sim in enumerate(sims):
        if sim < sim_threshold:
            # Check local minimum among neighbors
            left = sims[i - 1] if i > 0 else float("inf")
            right = sims[i + 1] if i < len(sims) - 1 else float("inf")
            if sim <= left and sim <= right:
                boundaries.add(i + 1)  # boundary before sentence i+1

    # Build initial chunks from boundaries
    chunks: list[Chunk] = []
    start = 0
    for boundary in sorted(boundaries):
        chunk_sents = sentences[start:boundary]
        chunks.append(Chunk(
            text=" ".join(chunk_sents),
            sentence_indices=list(range(start, boundary)),
        ))
        start = boundary
    # Final chunk
    if start < len(sentences):
        chunks.append(Chunk(
            text=" ".join(sentences[start:]),
            sentence_indices=list(range(start, len(sentences))),
        ))

    # Merge adjacent short chunks to approach target_chars
    merged: list[Chunk] = []
    acc_text: list[str] = []
    acc_indices: list[int] = []
    for c in chunks:
        acc_text.append(c.text)
        acc_indices.extend(c.sentence_indices)
        if sum(len(t) for t in acc_text) >= target_chars:
            merged.append(Chunk(text=" ".join(acc_text), sentence_indices=list(acc_indices)))
            acc_text = []
            acc_indices = []
    if acc_text:
        # Append leftover to last merged chunk or create new
        if merged:
            last = merged[-1]
            merged[-1] = Chunk(
                text=last.text + " " + " ".join(acc_text),
                sentence_indices=last.sentence_indices + acc_indices,
            )
        else:
            merged.append(Chunk(text=" ".join(acc_text), sentence_indices=acc_indices))

    return merged
