"""RAPTOR RAG baseline: recursive tree construction + multi-level retrieval.

Algorithm:
1. Flatten the document and split into Level 0 chunks.
2. Recursively cluster and summarize to build a hierarchical tree (up to max_depth levels).
3. At retrieval time, compute cosine similarity across all levels and take top-k.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from core.embed.embeddings import cosine_sim_batch, get_embeddings_batch, get_model_name_for_provider
from core.pipeline.e2e_utils.baselines import register_baseline
from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline
from core.pipeline.e2e_utils.rag_vanilla import chunk_text, embed_chunks, flatten_document

# Summary prompt template
_SUMMARY_PROMPT = "Summarize the following text passages concisely:\n\n{texts}\n\nSummary:"


@register_baseline("rag-raptor")
class RAPTORBaseline(BaseRAGBaseline):
    """RAPTOR RAG baseline: recursive clustering summary tree + multi-level vector retrieval."""

    name = "rag-raptor"

    def __init__(self, max_depth: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_depth = max_depth
        # Memory cache: {doc_id: (all_texts, all_embeddings)}
        self._mem_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def _cache_dir(self) -> Path:
        """Return the RAPTOR disk cache directory."""
        d = self.cache_root / "rag_raptor"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_tree(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        depth: int,
    ) -> tuple[list[str], np.ndarray]:
        """Recursively build a summary tree.

        Returns the concatenated (texts, embeddings) of the current and all parent levels.
        Input texts/embeddings are the current level; returns all newly added level nodes.
        """
        # Termination: too few chunks or maximum depth reached.
        if depth >= self.max_depth or len(texts) <= 1:
            return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

        n_clusters = min(max(2, len(texts) // 5), len(texts))

        # Degenerate case: cannot cluster.
        if n_clusters >= len(texts):
            return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

        # Try GMM; fall back to KMeans on failure.
        try:
            gm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=100)
            labels = gm.fit_predict(embeddings)
        except (ValueError, RuntimeError) as e:
            print(f"  GMM failed ({type(e).__name__}), falling back to KMeans", file=sys.stderr)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(embeddings)

        # Concatenate texts per cluster and generate a summary.
        summary_texts: list[str] = []
        for c_id in range(n_clusters):
            indices = [i for i, lbl in enumerate(labels) if lbl == c_id]
            if not indices:
                continue
            combined = "\n\n".join(texts[i] for i in indices)
            prompt = _SUMMARY_PROMPT.format(texts=combined)
            result = self.cached_caller.call(
                prompt,
                llm_provider=self.llm_provider,
                model=self.llm_model,
                max_tokens=300,
            )
            self.track_preprocess_call(result)
            summary_texts.append(result.response)

        if not summary_texts:
            return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

        # Embed summaries.
        model_name = get_model_name_for_provider(self.embed_provider)
        summary_embs = np.array(
            get_embeddings_batch(summary_texts, model=model_name, provider=self.embed_provider),
            dtype=np.float32,
        )

        # Recursively process higher-level summaries.
        parent_texts, parent_embs = self._build_tree(summary_texts, summary_embs, depth + 1)

        # Current-level summaries + higher-level summaries.
        all_new_texts = summary_texts + parent_texts
        if parent_embs.shape[0] > 0:
            all_new_embs = np.vstack([summary_embs, parent_embs])
        else:
            all_new_embs = summary_embs

        return all_new_texts, all_new_embs

    def preprocess_doc(self, doc_id: str) -> None:
        """Preprocess a single document: build the RAPTOR tree and cache to disk."""
        cache_dir = self._cache_dir()
        text_cache = cache_dir / f"{doc_id}.json"
        emb_cache = cache_dir / f"{doc_id}_embeddings.npz"

        # Skip if disk cache exists.
        if text_cache.exists() and emb_cache.exists():
            return

        # Load and split document.
        json_path = self.processing_dir / f"{doc_id}_reconstructed.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping", file=sys.stderr)
            return

        flat_text = flatten_document(json_path)
        level0_chunks = chunk_text(flat_text, self.chunk_size, self.chunk_overlap)
        if not level0_chunks:
            return

        # Level 0 embeddings (reuse embed_chunks with caching).
        level0_embs = embed_chunks(
            level0_chunks,
            self.embed_provider,
            doc_id,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        # Recursively build summary tree and collect all upper-level nodes.
        parent_texts, parent_embs = self._build_tree(level0_chunks, level0_embs, depth=0)

        # Merge nodes from all levels.
        all_texts = level0_chunks + parent_texts
        if parent_embs.shape[0] > 0:
            all_embs = np.vstack([level0_embs, parent_embs])
        else:
            all_embs = level0_embs

        # Write to disk cache.
        with open(text_cache, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, ensure_ascii=False)
        np.savez_compressed(emb_cache, embeddings=all_embs)

    def _load_cache(self, doc_id: str) -> tuple[list[str], np.ndarray] | None:
        """Load cached texts + embeddings from memory or disk."""
        if doc_id in self._mem_cache:
            return self._mem_cache[doc_id]

        cache_dir = self._cache_dir()
        text_cache = cache_dir / f"{doc_id}.json"
        emb_cache = cache_dir / f"{doc_id}_embeddings.npz"

        if not text_cache.exists() or not emb_cache.exists():
            return None

        with open(text_cache, "r", encoding="utf-8") as f:
            texts: list[str] = json.load(f)

        data = np.load(emb_cache)
        embs: np.ndarray = data["embeddings"]

        self._mem_cache[doc_id] = (texts, embs)
        return texts, embs

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k across nodes at all tree levels."""
        cached = self._load_cache(doc_id)
        if cached is None:
            return []

        texts, embs = cached
        if embs.shape[0] == 0:
            return []

        scores = cosine_sim_batch(embs, np.array(query_embedding, dtype=np.float32))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(texts[i], float(scores[i])) for i in top_indices]
