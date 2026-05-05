"""Search tool: embedding-based semantic chunk retrieval.

Embeddings are generated via a dedicated Azure OpenAI embedding endpoint.
Credentials are loaded from the file pointed to by ``embedding_key_file``
in ``local/azure.json`` (separate key/endpoint from the main LLM).

Computed embeddings are cached as JSON under ``embeddings/officeqa/`` so
subsequent runs skip recomputation.

Input:  text (raw string) or path (UTF-8 file) + keyword (query string)
Output: top-k chunks ranked by cosine similarity to the query embedding
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_EMBED_CACHE_DIR = _ROOT / "embeddings" / "officeqa"


def _load_embed_credentials() -> tuple[str, str, str, str]:
    """Load embedding credentials from the ``embedding_key_file`` in ``local/azure.json``.

    Returns ``(api_key, api_version, endpoint, deployment)``.
    """
    from azure_local import _parse_key_file_text

    cfg_path = _ROOT / "local" / "azure.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    emb_key_file = raw.get("embedding_key_file")
    if not emb_key_file:
        raise ValueError("embedding_key_file not set in local/azure.json")

    key_path = Path(emb_key_file).expanduser()
    parsed = _parse_key_file_text(key_path.read_text(encoding="utf-8"))

    api_key    = parsed.get("api_key", "")
    api_version = parsed.get("api_version", "2024-12-01-preview")
    endpoint   = parsed.get("endpoint", "").rstrip("/")
    deployment = parsed.get("deployment", "text-embedding-3-small")
    return api_key, api_version, endpoint, deployment


def _embed_client() -> Any:
    """Return an AzureOpenAI client configured for the embedding endpoint."""
    from openai import AzureOpenAI

    api_key, api_version, endpoint, _ = _load_embed_credentials()
    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)


def _embed_deployment() -> str:
    """Return the embedding deployment name from ``embedding_key_file``."""
    try:
        _, _, _, deployment = _load_embed_credentials()
        if deployment:
            return deployment
    except Exception:
        pass
    return "text-embedding-3-small"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cache_path(source_id: str) -> Path:
    _EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_id)[:80]
    return _EMBED_CACHE_DIR / f"{safe}.json"


def _load_cache(cache_file: Path) -> dict[str, Any]:
    if cache_file.is_file():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache_file: Path, data: dict[str, Any]) -> None:
    try:
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_embeddings_batch(texts: list[str], deployment: str) -> list[list[float]]:
    """Call Azure OpenAI to get embeddings for a list of texts."""
    client = _embed_client()
    response = client.embeddings.create(model=deployment, input=texts)
    return [d.embedding for d in response.data]


# ---------------------------------------------------------------------------
# Core logic (importable without LangChain)
# ---------------------------------------------------------------------------

def _split_chunks(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _keyword_rank(chunks: list[str], query: str, top_k: int) -> list[str]:
    """Fallback ranking by token overlap when embeddings are unavailable."""
    query_tokens = set(query.lower().split())
    scored = [(len(query_tokens & set(c.lower().split())) / max(len(query_tokens), 1), c)
              for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for score, c in scored[:top_k] if score > 0]
    return result if result else chunks[:top_k]


def search_passages(
    text: str,
    query: str,
    *,
    source_id: str = "",
    top_k: int = 5,
) -> list[str]:
    """Return the top-k chunks from ``text`` most similar to ``query``.

    Embeddings for ``source_id`` are cached on disk.  If the Azure embedding
    call fails the function falls back to token-overlap ranking.
    """
    chunks = _split_chunks(text)
    if not chunks:
        return []

    deployment = _embed_deployment()
    cache_file = _cache_path(source_id or hashlib.md5(text[:200].encode()).hexdigest())
    cache = _load_cache(cache_file)

    # Embed chunks (use cache when available)
    chunk_embeddings: list[list[float]] = []
    missing_indices: list[int] = []
    for i, chunk in enumerate(chunks):
        key = hashlib.md5(chunk.encode()).hexdigest()
        if key in cache:
            chunk_embeddings.append(cache[key])
        else:
            chunk_embeddings.append([])  # placeholder
            missing_indices.append(i)

    if missing_indices:
        batch_texts = [chunks[i] for i in missing_indices]
        try:
            batch_vecs = _get_embeddings_batch(batch_texts, deployment)
            for idx, vec in zip(missing_indices, batch_vecs):
                key = hashlib.md5(chunks[idx].encode()).hexdigest()
                cache[key] = vec
                chunk_embeddings[idx] = vec
            _save_cache(cache_file, cache)
        except Exception:
            return _keyword_rank(chunks, query, top_k)

    # Embed query
    try:
        query_vec = _get_embeddings_batch([query], deployment)[0]
    except Exception:
        return _keyword_rank(chunks, query, top_k)

    # Rank by cosine similarity
    similarities = [(_cosine(query_vec, vec), chunk) for vec, chunk in zip(chunk_embeddings, chunks)]
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in similarities[:top_k]]


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def search(text: str = "", keyword: str = "", path: str = "", top_k: int = 5) -> str:
    """Semantic (embedding) search over a document.

    Provide either ``path`` (a UTF-8 text file to load) or raw ``text``.
    Returns the top ``top_k`` paragraphs most semantically similar to
    ``keyword``, separated by '---'.  Embeddings are cached locally to avoid
    recomputation.
    """
    p = (path or "").strip()
    if p:
        fp = Path(p)
        if not fp.is_file():
            return f"(file not found: {p})"
        body = fp.read_text(encoding="utf-8", errors="replace")
        source_id = Path(p).stem
    else:
        body = text
        source_id = hashlib.md5(text[:200].encode()).hexdigest() if text else ""

    kw = (keyword or "").strip()
    if not kw:
        return "(no keyword provided)"

    hits = search_passages(body, kw, source_id=source_id, top_k=int(top_k))
    if not hits:
        return "(no results found)"
    return "\n---\n".join(hits)
