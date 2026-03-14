"""core.cluster.common -- Shared utilities across phases."""

from core.cluster.common.io import extract_headers, get_doc_ids, load_json
from core.cluster.common.paths import (
    CHI_EMBEDDING,
    CHI_PROCESSING,
    D_EMB,
    D_POS,
    D_VIS,
    SEC_EMBEDDING,
    SEC_PROCESSING,
    SEED,
    TYPE_ORDER_4,
    TYPE_ORDER_5,
    TYPE_ORDER_FULL,
    TYPE_ORDER_MERGED,
    get_embedding_dir,
    get_processing_dir,
)

__all__ = [
    # io
    "load_json",
    "get_doc_ids",
    "extract_headers",
    # paths
    "SEED",
    "SEC_PROCESSING",
    "CHI_PROCESSING",
    "SEC_EMBEDDING",
    "CHI_EMBEDDING",
    "D_EMB",
    "D_POS",
    "D_VIS",
    "TYPE_ORDER_4",
    "TYPE_ORDER_5",
    "TYPE_ORDER_FULL",
    "TYPE_ORDER_MERGED",
    "get_processing_dir",
    "get_embedding_dir",
]
