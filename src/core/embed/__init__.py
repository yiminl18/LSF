# -*- coding: utf-8 -*-
"""Embeddings and vectors: unified embedding interface, caching, and Azure client."""

from core.embed.embeddings import (
    get_embedding,
    get_embeddings_batch,
    get_query_embedding,
    load_document_embeddings,
    build_embedding,
    cosine_sim,
    get_combined_text,
    get_model_name_for_provider,
)

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "get_query_embedding",
    "load_document_embeddings",
    "build_embedding",
    "cosine_sim",
    "get_combined_text",
    "get_model_name_for_provider",
]
