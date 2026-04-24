"""rag-vanilla baseline adapter.

Wraps the functions from rag_vanilla.py into the BaseRAGBaseline interface.
All original functions remain in rag_vanilla.py; this module is a thin adapter only.
"""

import numpy as np

from core.pipeline.e2e_utils.baselines import register_baseline
from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline
from core.pipeline.e2e_utils.rag_vanilla import (
    embed_chunks,
    get_chunks_cached,
    retrieve_chunks,
)


@register_baseline("rag-vanilla")
class VanillaRAGBaseline(BaseRAGBaseline):
    """Standard RAG baseline: flatten document → chunk → embedding retrieval."""

    name = "rag-vanilla"

    def preprocess_doc(self, doc_id: str) -> None:
        """Pre-load chunk text and embedding cache (disk + memory)."""
        chunks = get_chunks_cached(
            doc_id,
            self.processing_dir,
            chunk_size_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
        )
        if chunks:
            embed_chunks(
                chunks,
                self.embed_provider,
                doc_id,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k chunks using cached chunk embeddings."""
        chunks = get_chunks_cached(
            doc_id,
            self.processing_dir,
            chunk_size_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
        )
        if not chunks:
            return []

        # embed_chunks reads from memory/disk cache — no redundant API calls.
        chunk_embs: np.ndarray = embed_chunks(
            chunks,
            self.embed_provider,
            doc_id,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        return retrieve_chunks(query_embedding, chunk_embs, chunks, top_k=top_k)
