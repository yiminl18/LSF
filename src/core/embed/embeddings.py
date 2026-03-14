"""
Embeddings Module

Unified module for generating, caching, and managing text embeddings.
Supports OpenAI/Azure/OpenRouter API.

Main functions:
- get_embedding(): Get embedding for a single text.
- get_embeddings_batch(): Get embeddings for a list of texts.
- get_query_embedding(): Get embedding for a query (cached).
- build_embedding(): Build and cache embeddings for a document (merged JSON).
- load_document_embeddings(): Load cached embeddings for a document.
- cosine_sim(): Calculate cosine similarity.

Dependencies:
- openai
- sklearn
- numpy
- torch
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import tiktoken
from openai import AzureOpenAI, OpenAI
from core.utils.progress import rich_tqdm as tqdm

logger = logging.getLogger(__name__)
from core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_MILLION,
    EMBEDDING_API_MAX_INPUT_TOKENS,
)

# Global mutable state: embedding cost tracking (process-level, not thread-safe)
# Access via reset_embedding_cost() / get_embedding_cost()
_cumulative_embedding_cost_usd = 0.0
_embedding_total_tokens = 0

# Global mutable state: in-process caches (not thread-safe)
# _query_embedding_caches: per-provider query vector cache to avoid redundant API calls
_query_embedding_caches: Dict[str, Dict[str, List[float]]] = {}
_DEFAULT_OPENAI_BATCH_SIZE = 100
_API_PROVIDERS = {"openai", "azure", "openrouter"}


def reset_embedding_cost() -> None:
    """Reset the embedding cost counter."""
    global _cumulative_embedding_cost_usd, _embedding_total_tokens
    _cumulative_embedding_cost_usd = 0.0
    _embedding_total_tokens = 0


def get_embedding_cost() -> Tuple[int, float]:
    """
    Get the embedding cost so far.

    Returns:
        (total_tokens, cost_usd)
    """
    return _embedding_total_tokens, _cumulative_embedding_cost_usd


def get_model_name_for_provider(provider: str) -> str:
    """
    Get the model name for a given provider.

    Args:
        provider: embedding provider (openai, azure, openrouter)

    Returns:
        Corresponding model name string
    """
    mapping = {
        "openai": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "azure": DEFAULT_EMBEDDING_MODEL,
        "openrouter": os.environ.get(
            "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        ),
    }
    return mapping.get(provider, DEFAULT_EMBEDDING_MODEL)


def resolve_embedding_batch_size(provider: str, batch_size: Optional[int]) -> int:
    """Resolve the batch size for the given provider."""
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        return batch_size
    return _DEFAULT_OPENAI_BATCH_SIZE


def _create_api_embedding_client(provider: str) -> Any:
    """Create an API embedding client."""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for openai embedding provider.")
        return OpenAI(api_key=api_key)
    if provider == "azure":
        api_key = os.environ.get("AZURE_EMBEDDING_API_KEY")
        azure_endpoint = os.environ.get("AZURE_EMBEDDING_API_BASE")
        api_version = os.environ.get("AZURE_API_VERSION")
        if not all([api_key, azure_endpoint, api_version]):
            raise ValueError("Missing AZURE_EMBEDDING env vars.")
        return AzureOpenAI(
            azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
        )
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OPENROUTER_API_KEY for openrouter embedding provider."
            )
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    raise ValueError(f"Unknown API provider: {provider}")


def _resolve_token_encoder(model: str):
    """Resolve token encoder, falling back to cl100k_base on failure."""
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        if "/" in model:
            try:
                return tiktoken.encoding_for_model(model.split("/", 1)[1])
            except Exception:
                pass
        return tiktoken.get_encoding("cl100k_base")


def resolve_api_embedding_max_input_tokens(provider: str) -> int:
    """Resolve the max input token limit for API embedding requests."""
    provider_env = f"{provider.upper()}_EMBEDDING_MAX_INPUT_TOKENS"
    raw = os.environ.get(provider_env) or os.environ.get(
        "LSF_EMBEDDING_MAX_INPUT_TOKENS"
    )
    if raw is None:
        return EMBEDDING_API_MAX_INPUT_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid token limit '{raw}' from {provider_env}/LSF_EMBEDDING_MAX_INPUT_TOKENS."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Token limit must be > 0, got {value} from {provider_env}/LSF_EMBEDDING_MAX_INPUT_TOKENS."
        )
    return value


def _truncate_text_by_tokens(
    text: str,
    encoder: Any,
    max_tokens: int,
) -> Tuple[str, bool]:
    """Truncate text by token count. Returns (truncated_text, was_truncated)."""
    token_ids = encoder.encode(text)
    if len(token_ids) <= max_tokens:
        return text, False
    truncated = encoder.decode(token_ids[:max_tokens]).strip()
    if not truncated:
        # Edge case: if decode returns empty string, fall back to character-level truncation
        truncated = text[: max(1, min(len(text), 1024))]
    return truncated, True


def _extract_response_total_tokens(
    response: Any, batch_texts: List[str], model: str
) -> int:
    """Prefer API usage.total_tokens; estimate from input text if unavailable."""
    usage = getattr(response, "usage", None)
    if usage:
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            return int(total_tokens)

    encoder = _resolve_token_encoder(model)
    return int(sum(len(encoder.encode(text)) for text in batch_texts))


def _get_query_cache_path(source: str = "pdfs", provider: str = "openai") -> Path:
    from core.utils.paths import PathManager

    paths = PathManager()
    return paths.get_query_embedding_path(dataset=source, provider=provider)


def _load_query_embedding_cache(source: str, provider: str) -> Dict[str, List[float]]:
    cache_key = f"{source}_{provider}"
    if cache_key in _query_embedding_caches:
        return _query_embedding_caches[cache_key]

    cache_path = _get_query_cache_path(source, provider)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logger.warning("Failed to load query cache from %s: %s", cache_path, e)

    _query_embedding_caches[cache_key] = cache
    return cache


def _save_query_embedding_cache(source: str, provider: str) -> None:
    cache_key = f"{source}_{provider}"
    if cache_key not in _query_embedding_caches:
        return

    cache = _query_embedding_caches[cache_key]
    cache_path = _get_query_cache_path(source, provider)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_embeddings_batch(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
    provider: str = "openai",
    max_retries: int = 3,
) -> List[List[float]]:
    """Batch generate embeddings with all-zero vector detection and retry."""
    if not texts:
        return []

    cleaned_texts = []
    empty_indices = set()
    for i, text in enumerate(texts):
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            empty_indices.add(i)
        cleaned_texts.append(cleaned)

    non_empty = [(i, t) for i, t in enumerate(cleaned_texts) if i not in empty_indices]
    dim = 1536

    if not non_empty:
        return [[0.0] * dim for _ in texts]

    non_empty_texts = [t for _, t in non_empty]
    non_empty_indices = [i for i, _ in non_empty]
    request_texts = list(non_empty_texts)
    resolved_batch_size = resolve_embedding_batch_size(provider, batch_size)

    if provider in _API_PROVIDERS:
        encoder = _resolve_token_encoder(model)
        max_tokens = resolve_api_embedding_max_input_tokens(provider)
        truncated_count = 0
        for idx, text in enumerate(request_texts):
            truncated_text, is_truncated = _truncate_text_by_tokens(
                text, encoder, max_tokens
            )
            request_texts[idx] = truncated_text
            if is_truncated:
                truncated_count += 1
        if truncated_count > 0:
            logger.warning(
                "Truncated %d/%d texts to %d tokens for provider=%s model=%s",
                truncated_count,
                len(request_texts),
                max_tokens,
                provider,
                model,
            )

    results = [None] * len(texts)
    to_retry = list(range(len(request_texts)))

    for attempt in range(max_retries):
        current_texts = [request_texts[i] for i in to_retry]
        current_indices = [non_empty_indices[i] for i in to_retry]

        if not current_texts:
            break

        batch_results = [None] * len(current_texts)

        if provider in _API_PROVIDERS:
            client = _create_api_embedding_client(provider)
            iterator = range(0, len(current_texts), resolved_batch_size)
            if show_progress and attempt == 0:
                iterator = tqdm(
                    iterator, desc="Embeddings API", unit="batch", leave=False
                )

            for start in iterator:
                batch_texts = current_texts[start : start + resolved_batch_size]
                batch_indices = current_indices[start : start + resolved_batch_size]
                local_batch_positions = list(range(start, start + len(batch_texts)))
                try:
                    global _cumulative_embedding_cost_usd, _embedding_total_tokens
                    response = client.embeddings.create(input=batch_texts, model=model)
                    tokens = _extract_response_total_tokens(
                        response, batch_texts, model
                    )
                    _embedding_total_tokens += tokens
                    _cumulative_embedding_cost_usd += tokens * (
                        EMBEDDING_PRICE_PER_MILLION / 1_000_000
                    )
                    for j, item in enumerate(response.data):
                        embedding = item.embedding
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                        batch_results[local_batch_positions[j]] = embedding
                except Exception as e:
                    logger.warning(
                        "Batch embedding failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        e,
                    )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Process results
        new_to_retry = []
        for j, (non_empty_idx, text) in enumerate(zip(to_retry, current_texts)):
            emb = batch_results[j]
            if emb is not None and not is_all_zero_vector(emb):
                results[non_empty_indices[non_empty_idx]] = emb
            else:
                if attempt < max_retries - 1:
                    new_to_retry.append(non_empty_idx)
                    logger.warning(
                        "All-zero vector detected (attempt %d/%d), retrying for text: %s",
                        attempt + 1,
                        max_retries,
                        text[:100],
                    )
                else:
                    logger.warning(
                        "All-zero vector generated after %d attempts, skipping text: %s",
                        max_retries,
                        text[:100],
                    )
                    results[non_empty_indices[non_empty_idx]] = [0.0] * dim

        to_retry = new_to_retry

    for i in empty_indices:
        results[i] = [0.0] * dim
    return results


def is_all_zero_vector(vec: Any) -> bool:
    """Check if a vector is all zeros or contains only NaNs."""
    if vec is None:
        return True
    if isinstance(vec, np.ndarray):
        if np.any(np.isnan(vec)):
            return True
        return np.all(vec == 0.0)
    if isinstance(vec, list):
        for v in vec:
            if v is None:
                return True
            if isinstance(v, float):
                if np.isnan(v):
                    return True
            else:
                try:
                    if np.isnan(float(v)):
                        return True
                except (ValueError, TypeError):
                    pass
        return all(v == 0.0 for v in vec)
    return False


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = "openai",
    max_retries: int = 3,
) -> list:
    """Get embedding for a single text with all-zero vector detection and retry."""
    text = text.replace("\n", " ").strip()
    request_text = text
    dim = 1536

    if not text:
        return [0.0] * dim

    for attempt in range(max_retries):
        if provider in _API_PROVIDERS:
            try:
                global _cumulative_embedding_cost_usd, _embedding_total_tokens
                encoder = _resolve_token_encoder(model)
                max_tokens = resolve_api_embedding_max_input_tokens(provider)
                request_text, is_truncated = _truncate_text_by_tokens(
                    text, encoder, max_tokens
                )
                if is_truncated and attempt == 0:
                    logger.warning(
                        "Truncated single text to %d tokens for provider=%s model=%s",
                        max_tokens,
                        provider,
                        model,
                    )
                client = _create_api_embedding_client(provider)
                response = client.embeddings.create(input=[request_text], model=model)
                # Cost tracking
                tokens = _extract_response_total_tokens(response, [request_text], model)
                _embedding_total_tokens += tokens
                _cumulative_embedding_cost_usd += tokens * (
                    EMBEDDING_PRICE_PER_MILLION / 1_000_000
                )
                emb = response.data[0].embedding
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                elif not isinstance(emb, list):
                    emb = list(emb)
            except Exception as e:
                logger.warning(
                    "Failed to get embedding (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt == max_retries - 1:
                    logger.warning(
                        "All-zero vector generated after %d attempts, skipping text: %s",
                        max_retries,
                        text[:100],
                    )
                    return [0.0] * dim
                continue
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not is_all_zero_vector(emb):
            return emb

        if attempt < max_retries - 1:
            logger.warning(
                "All-zero vector detected (attempt %d/%d), retrying for text: %s",
                attempt + 1,
                max_retries,
                text[:100],
            )

    logger.warning(
        "All-zero vector generated after %d attempts, skipping text: %s",
        max_retries,
        text[:100],
    )
    return [0.0] * dim


def get_query_embedding(
    query: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = "openai",
    source: str = "pdfs",
) -> List[float]:
    """Get query embedding from cache or calculate."""
    cache = _load_query_embedding_cache(source, provider)
    if query in cache:
        return cache[query]

    current_model = model
    embedding = get_embedding(query, model=current_model, provider=provider)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    elif not isinstance(embedding, list):
        embedding = list(embedding)

    cache[query] = embedding
    _save_query_embedding_cache(source, provider)

    # Check if query embedding is a zero vector (usually indicates API misconfiguration or empty text)
    norm = np.linalg.norm(np.asarray(embedding, dtype=np.float32))
    if norm < 1e-6:
        logger.warning(
            f"Query embedding norm ≈ 0 (norm={norm:.2e}), provider={provider}, model={current_model!r}. "
            "Possibly a DEFAULT_EMBEDDING_MODEL misconfiguration or the API returned a zero vector."
        )

    return embedding


def cosine_sim(vec1, vec2) -> float:
    """Compute cosine similarity of two vectors. Accepts list or np.ndarray; zero-copy when already ndarray."""
    if vec1 is None or vec2 is None:
        return 0.0
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    result = np.dot(v1, v2) / (norm1 * norm2)
    if np.isnan(result):
        return 0.0
    return float(np.clip(result, -1.0, 1.0))


def cosine_sim_batch(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Batch compute cosine similarity between each row of a matrix and a vector.

    Args:
        matrix: (N, D) embedding matrix
        vector: (D,) query vector

    Returns:
        (N,) similarity array
    """
    if matrix.size == 0 or vector.size == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    # Clear inf/nan values (copy=False is safe here since we already cast to float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    norm_v = np.linalg.norm(vector)
    if norm_v == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        norms_m = np.linalg.norm(matrix, axis=1)
        dots = matrix @ vector
        # Zero-vector rows return 0
        safe_norms = np.where(norms_m > 0, norms_m, 1.0)
        sims = dots / (safe_norms * norm_v)
    sims[norms_m == 0] = 0.0
    np.nan_to_num(sims, copy=False, nan=0.0)
    return np.clip(sims, -1.0, 1.0)


def build_header_embedding_matrix(header_list, embeddings: Dict[str, Any], dim: int):
    """Build dense matrices and validity masks from header_list and embeddings dict.

    Builds one set of matrices each for combined_text and processing_path.

    Args:
        header_list: list of HeaderNode objects (must have combined_text / processing_path attributes)
        embeddings: {text: list[float]} dictionary
        dim: embedding dimension

    Returns:
        (emb_matrix, emb_valid, path_matrix, path_valid)
        All shapes are (N, dim) / (N,)
    """
    n = len(header_list)
    emb_mat = np.zeros((n, dim), dtype=np.float32)
    emb_valid = np.zeros(n, dtype=bool)
    path_mat = np.zeros((n, dim), dtype=np.float32)
    path_valid = np.zeros(n, dtype=bool)

    for i, h in enumerate(header_list):
        e = embeddings.get(h.combined_text)
        if e is not None:
            emb_mat[i] = (
                e if isinstance(e, np.ndarray) else np.asarray(e, dtype=np.float32)
            )
            emb_valid[i] = True
        p = h.processing_path
        pe = embeddings.get(p) if p else None
        if pe is not None:
            path_mat[i] = (
                pe if isinstance(pe, np.ndarray) else np.asarray(pe, dtype=np.float32)
            )
            path_valid[i] = True

    return emb_mat, emb_valid, path_mat, path_valid


# -----------------------------------------------------------------------------
# Document Embedding Logic (from get_provenance_node.py)
# -----------------------------------------------------------------------------


def get_combined_text(header: Dict[str, Any]) -> str:
    """Merge path and body text. path_text contains the full ancestor chain, providing hierarchical context."""
    structure = header.get("structure") or {}
    path_text = structure.get("path_text", "") or header.get("text", "")
    text_span = header.get("text_span", "")
    return f"{path_text} {text_span}".strip()


def _save_embeddings_npz(path: Path, emb_dict: Dict[str, Any]) -> None:
    """Save embedding dict as npz (float32, saves space and load time)."""
    if not emb_dict:
        return
    keys = list(emb_dict.keys())
    values = np.array([emb_dict[k] for k in keys], dtype=np.float32)
    # Use unicode array for keys to avoid object pickle deserialization
    np.savez(path, keys=np.asarray(keys, dtype=np.str_), values=values)


def _load_embeddings_npz_core(path: Path, allow_pickle: bool) -> Dict[str, np.ndarray]:
    """Core implementation for loading embedding dict from npz."""
    with np.load(path, allow_pickle=allow_pickle) as data:
        keys = data["keys"]
        values = data["values"]  # float32 matrix
        return {str(k): values[i] for i, k in enumerate(keys)}


def _load_embeddings_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load embedding dict from npz."""
    try:
        return _load_embeddings_npz_core(path, allow_pickle=False)
    except Exception:
        # Compatible with legacy object-key npz files (requires allow_pickle=True)
        return _load_embeddings_npz_core(path, allow_pickle=True)


def build_embedding(
    merged_json_path: str,
    cache_dir: Optional[str] = None,
    provider: str = "openai",
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    dataset: str = "pdfs",
) -> Tuple[int, int]:
    """
    Build and cache embeddings for a merged JSON file.
    Only computes new embeddings for text not already in cache.
    Saved in npz format.

    Returns:
        (total_embeddings, skipped_all_zero)
    """
    merged_json_file = Path(merged_json_path)

    if cache_dir:
        cache_directory = Path(cache_dir)
    else:
        from core.utils.paths import PathManager

        paths = PathManager()
        cache_directory = paths.get_embeddings_dir(dataset, provider)

    cache_directory.mkdir(parents=True, exist_ok=True)
    npz_path = cache_directory / f"{merged_json_file.stem}_embeddings.npz"

    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    texts = merged_data.get("texts", [])
    # reconstructed.json contains many body nodes; embeddings only need section_headers to avoid key explosion
    from core.config import MAX_SPAN_WORDS

    headers = [
        t
        for t in texts
        if isinstance(t, dict)
        and (t.get("label") in (None, "section_header"))
        and len(t.get("text_span", "").split()) < MAX_SPAN_WORDS
    ]
    if not headers:
        if show_progress:
            print("No headers found.")
        return 0, 0

    current_texts = []
    seen = set()
    for header in headers:
        combined_text = get_combined_text(header)
        if combined_text and combined_text not in seen:
            current_texts.append(combined_text)
            seen.add(combined_text)
        path_text = (header.get("structure") or {}).get("path_text", "") or header.get(
            "path_text", ""
        )
        if path_text and path_text not in seen:
            current_texts.append(path_text)
            seen.add(path_text)

    # Load existing cache (npz)
    cached_embeddings: Dict[str, Any] = {}
    all_zero_count = 0
    if npz_path.exists():
        try:
            cached_embeddings = _load_embeddings_npz(npz_path)
            # Check and remove all-zero vectors
            keys_to_remove = []
            for key, vec in cached_embeddings.items():
                if is_all_zero_vector(vec):
                    keys_to_remove.append(key)
                    all_zero_count += 1
            for key in keys_to_remove:
                del cached_embeddings[key]
            if show_progress and all_zero_count > 0:
                print(
                    f"Loaded {len(cached_embeddings)} cached embeddings (removed {all_zero_count} all-zero vectors)."
                )
            elif show_progress:
                print(f"Loaded {len(cached_embeddings)} cached embeddings.")
        except Exception:
            cached_embeddings = {}

    cached_keys = set(cached_embeddings.keys())
    to_add_list = [t for t in current_texts if t not in cached_keys]

    if not to_add_list:
        if show_progress:
            print(f"Cache ({provider}) is up-to-date.")
        return len(cached_embeddings), all_zero_count

    if show_progress:
        print(f"Adding {len(to_add_list)} new embeddings...")
    model = get_model_name_for_provider(provider)

    # Incremental save: process in batches, save after each batch to support resumption
    resolved_batch_size = resolve_embedding_batch_size(provider, batch_size)
    total_batches = (len(to_add_list) + resolved_batch_size - 1) // resolved_batch_size

    total_skipped_all_zero = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * resolved_batch_size
        end_idx = min(start_idx + resolved_batch_size, len(to_add_list))
        batch_texts = to_add_list[start_idx:end_idx]

        if show_progress and total_batches > 1:
            print(
                f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} texts)..."
            )

        try:
            batch_embeddings = get_embeddings_batch(
                batch_texts,
                model=model,
                batch_size=resolved_batch_size,
                provider=provider,
                show_progress=False,  # Avoid nested progress bars
            )

            # Add to cache
            for combined_text, embedding_vector in zip(batch_texts, batch_embeddings):
                if hasattr(embedding_vector, "tolist"):
                    embedding_vector = embedding_vector.tolist()
                elif not isinstance(embedding_vector, list):
                    embedding_vector = list(embedding_vector)

                if is_all_zero_vector(embedding_vector):
                    total_skipped_all_zero += 1
                    logger.warning(
                        "Skipping all-zero vector for text: %s", combined_text[:100]
                    )
                else:
                    cached_embeddings[combined_text] = embedding_vector

            # Save immediately to support resumption
            _save_embeddings_npz(npz_path, cached_embeddings)

            if show_progress and total_batches > 1:
                print(
                    f"Saved batch {batch_idx + 1}/{total_batches} ({len(cached_embeddings)} total embeddings)"
                )
        except Exception:
            # If a batch fails, save completed batches and re-raise the exception
            if cached_embeddings:
                _save_embeddings_npz(npz_path, cached_embeddings)
                if show_progress:
                    print(
                        f"Saved {len(cached_embeddings)} embeddings before error. You can resume later."
                    )
            raise

    total_skipped = all_zero_count + total_skipped_all_zero
    if total_skipped > 0:
        print(f"\n{'=' * 60}")
        print(f"WARNING: Skipped {total_skipped} all-zero vectors in this document!")
        print(f"{'=' * 60}\n")

    return len(cached_embeddings), total_skipped


def load_document_embeddings(
    merged_json_path: str,
    cache_dir: Optional[str] = None,
    provider: str = "openai",
    dataset: str = "pdfs",
) -> Tuple[Dict[str, Any], Path]:
    """Load document embeddings, building them if necessary.
    Uses npz cache; saves as npz when building."""
    if cache_dir:
        cache_directory = Path(cache_dir)
    else:
        from core.utils.paths import PathManager

        paths = PathManager()
        cache_directory = paths.get_embeddings_dir(dataset, provider)

    cache_directory.mkdir(parents=True, exist_ok=True)
    merged_json_file = Path(merged_json_path)

    npz_path = cache_directory / f"{merged_json_file.stem}_embeddings.npz"
    if not npz_path.exists():
        total_emb, skipped = build_embedding(
            merged_json_path, str(cache_directory), provider=provider
        )
        if skipped > 0:
            logger.debug(
                f"Skipped {skipped} all-zero vectors while building embeddings for {merged_json_path}"
            )
    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding cache not found after build: {npz_path}")

    embeddings_dict = _load_embeddings_npz(npz_path)

    # Check for zero vectors after loading (may come from historical bad cache)
    if embeddings_dict:
        norms = np.linalg.norm(
            np.stack(list(embeddings_dict.values())).astype(np.float32), axis=1
        )
        zero_count = int(np.sum(norms < 1e-6))
        if zero_count > 0:
            logger.warning(
                f"Document embeddings contain {zero_count}/{len(embeddings_dict)} zero vectors "
                f"(source: {npz_path.name}). Likely stale bad cache; consider deleting the npz file and regenerating."
            )

    return embeddings_dict, npz_path
