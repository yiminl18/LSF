"""
rag-vanilla baseline: flatten document → chunk → embedding retrieval → token budget truncation.

Standard RAG pipeline used as a comparison against LSF's structured retrieval.
Context token budget is aligned with xgb-v5 (context-level match).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken

from core.embed.embeddings import (
    cosine_sim_batch,
    get_embeddings_batch,
    get_model_name_for_provider,
)
from core.llm.tokens import estimate_tokens
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.llm.cost import compute_cost
from core.pipeline.e2e_utils.generation import GenerationResult, build_sections

# tiktoken encoder (reused globally)
_ENC = tiktoken.encoding_for_model("gpt-4o")

# chunk embedding cache: {doc_id: np.ndarray (N, D)}
_CHUNK_EMB_CACHE: dict[str, np.ndarray] = {}

# chunk text cache: {doc_id: list[str]}
_CHUNK_TEXT_CACHE: dict[str, list[str]] = {}

# rag-vanilla dedicated prompt template
_VANILLA_PROMPT_TEMPLATE = """\
Answer the question using ONLY the provided context from a document.
If the answer cannot be found in the context, respond with "Information not found."

Context:
---
{context}
---

Question: {question}
Answer:"""


def flatten_document(processing_json_path: Path) -> str:
    """Flatten a reconstructed JSON file into plain text.

    Skips empty entries; text_span is omitted when empty.
    """
    with open(processing_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = []
    for entry in data.get("texts", []):
        text = (entry.get("text") or "").strip()
        text_span = (entry.get("text_span") or "").strip()
        if not text and not text_span:
            continue
        if text_span:
            parts.append(f"{text}\n{text_span}")
        else:
            parts.append(text)

    return "\n".join(parts)


def chunk_text(
    text: str, chunk_size_tokens: int = 200, overlap_tokens: int = 0
) -> list[str]:
    """Split text into fixed-size token chunks.

    Args:
        text: input plain text
        chunk_size_tokens: tokens per chunk
        overlap_tokens: overlap tokens between consecutive chunks

    Returns:
        List of chunk strings.
    """
    if not text.strip():
        return []

    tokens = _ENC.encode(text)
    if len(tokens) == 0:
        return []

    step = max(chunk_size_tokens - overlap_tokens, 1)
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_str = _ENC.decode(tokens[start:end])
        if chunk_str.strip():
            chunks.append(chunk_str)
        if end >= len(tokens):
            break

    return chunks


_CHUNK_DISK_CACHE_DIR = Path(".cache/chunk_embeddings")


def embed_chunks(
    chunks: list[str],
    provider: str,
    doc_id: str,
    chunk_size: int = 200,
    overlap: int = 0,
) -> np.ndarray:
    """Embed a list of chunks with two-level cache (memory + disk).

    Disk cache path: .cache/chunk_embeddings/{provider}/{doc_id}_c{chunk_size}_o{overlap}.npz
    Cache key includes chunk parameters; changing parameters triggers re-embedding.

    Returns:
        (N, D) ndarray
    """
    cache_key = f"{doc_id}|{provider}|{chunk_size}|{overlap}"

    # 1. Memory cache
    if cache_key in _CHUNK_EMB_CACHE:
        return _CHUNK_EMB_CACHE[cache_key]

    if not chunks:
        empty = np.zeros((0, 1), dtype=np.float32)
        _CHUNK_EMB_CACHE[cache_key] = empty
        return empty

    # 2. Disk cache
    disk_dir = _CHUNK_DISK_CACHE_DIR / provider
    disk_path = disk_dir / f"{doc_id}_c{chunk_size}_o{overlap}.npz"
    if disk_path.exists():
        data = np.load(disk_path)
        embeddings = data["embeddings"]
        if embeddings.shape[0] == len(chunks):
            _CHUNK_EMB_CACHE[cache_key] = embeddings
            return embeddings
        # chunk count mismatch (parameter changed) — re-embed

    # 3. API call (same model and batch+anti-zero strategy as document embeddings)
    model_name = get_model_name_for_provider(provider)
    result = get_embeddings_batch(chunks, model=model_name, provider=provider)
    embeddings = np.array(result, dtype=np.float32)

    # Write to disk cache.
    disk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(disk_path, embeddings=embeddings)

    _CHUNK_EMB_CACHE[cache_key] = embeddings
    return embeddings


def retrieve_chunks(
    query_embedding: list[float],
    chunk_embeddings: np.ndarray,
    chunks: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Retrieve top-k chunks by cosine similarity."""
    if chunk_embeddings.shape[0] == 0:
        return []

    scores = cosine_sim_batch(chunk_embeddings, np.array(query_embedding))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(chunks[i], float(scores[i])) for i in top_indices]


def compute_ref_context_budget(
    ref_detail: dict,
    doc_id: str,
    eval_method: str,
) -> Optional[int]:
    """Compute context token budget from an xgb-v5 E2E detail entry.

    Reconstructs the xgb-v5 context sections string and counts its tokens.
    Returns the context-level token budget (excluding prompt chrome).
    """
    # Find the target doc in eval_ranking_details.
    target_docs = ref_detail.get("eval_ranking_details", {}).get(
        "target_docs_details", []
    )
    doc_entry = None
    for d in target_docs:
        if d.get("doc_id") == doc_id:
            doc_entry = d
            break

    if doc_entry is None:
        return None

    methods = doc_entry.get("methods", {})
    if eval_method not in methods:
        return None

    top_5 = methods[eval_method].get("top_5", [])

    # Read top_k_used from per-doc results (if available).
    top_k_used = len(top_5)
    for doc_res in ref_detail.get("documents", []):
        if doc_res.get("document") == doc_id:
            top_k_used = doc_res.get("top_k_used", len(top_5))
            break

    nodes = top_5[:top_k_used]
    if not nodes:
        return None

    return estimate_tokens(build_sections(nodes))


def build_context_with_budget(
    ranked_chunks: list[tuple[str, float]],
    context_budget: int,
) -> tuple[str, int]:
    """Concatenate ranked chunks until the context token budget is exhausted.

    Args:
        ranked_chunks: (chunk_text, score) in descending relevance order
        context_budget: context-level token limit

    Returns:
        (context_string, actual_chunks_used)
    """
    if not ranked_chunks or context_budget <= 0:
        return "", 0

    parts = []
    total_tokens = 0
    chunks_used = 0

    for chunk_text, _score in ranked_chunks:
        chunk_tokens = len(_ENC.encode(chunk_text))
        separator_tokens = len(_ENC.encode("\n---\n")) if parts else 0

        if total_tokens + separator_tokens + chunk_tokens > context_budget:
            # Partially truncate the last chunk to fit the budget.
            remaining = context_budget - total_tokens - separator_tokens
            if remaining > 0:
                partial_tokens = _ENC.encode(chunk_text)[:remaining]
                partial_text = _ENC.decode(partial_tokens)
                if partial_text.strip():
                    parts.append(partial_text)
                    chunks_used += 1
            break

        parts.append(chunk_text)
        total_tokens += separator_tokens + chunk_tokens
        chunks_used += 1

    return "\n---\n".join(parts), chunks_used


def generate_answer_from_text(
    question: str,
    context: str,
    cached_caller: CachedLLMCaller,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    max_tokens: int = 500,
) -> GenerationResult:
    """Generate an answer from plain-text context (rag-vanilla only)."""
    prompt = _VANILLA_PROMPT_TEMPLATE.format(context=context, question=question)

    call_kwargs = {
        "llm_provider": llm_provider,
        "max_tokens": max_tokens,
        "model": llm_model,
    }
    cache_result = cached_caller.call(prompt, **call_kwargs)

    cost_usd = compute_cost(
        cache_result.input_tokens,
        cache_result.output_tokens,
        llm_provider,
        model=llm_model,
    )

    return GenerationResult(
        answer=cache_result.response,
        input_tokens=cache_result.input_tokens,
        output_tokens=cache_result.output_tokens,
        latency_ms=cache_result.latency_ms,
        cost_usd=cost_usd,
        cache_hit=cache_result.cache_hit,
    )


def get_chunks_cached(
    doc_id: str,
    processing_dir: Path,
    chunk_size_tokens: int = 200,
    overlap_tokens: int = 0,
) -> list[str]:
    """Get chunks for a document (cached to avoid repeated flatten+chunk)."""
    cache_key = f"{doc_id}|{chunk_size_tokens}|{overlap_tokens}"
    if cache_key in _CHUNK_TEXT_CACHE:
        return _CHUNK_TEXT_CACHE[cache_key]

    json_path = processing_dir / f"{doc_id}_reconstructed.json"
    if not json_path.exists():
        _CHUNK_TEXT_CACHE[cache_key] = []
        return []

    text = flatten_document(json_path)
    chunks = chunk_text(text, chunk_size_tokens, overlap_tokens)
    _CHUNK_TEXT_CACHE[cache_key] = chunks
    return chunks
