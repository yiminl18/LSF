"""core.cluster.common.paths -- Path constants and directory resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

# -- Shared constants --
SEED = 42

# -- Dataset directories --
SEC_PROCESSING = Path("datasets/pdfs/latest/processing-newer")
CHI_PROCESSING = Path("datasets/paper/latest/processing")
SEC_EMBEDDING = Path("datasets/pdfs/latest/embedding/openrouter/document_embedding")
CHI_EMBEDDING = Path("datasets/paper/latest/embedding/openrouter/document_embedding")
DEFAULT_EMBED_PROVIDER = "openrouter"

# -- Embedding dimensions --
D_EMB = 1536
D_POS = 6
D_VIS = 6

# -- Document type orderings --
TYPE_ORDER_4 = ["10K", "10Q", "8K", "EARNINGS"]  # SEC 4-class (ANNUAL merged)
TYPE_ORDER_5 = ["10K", "10Q", "8K", "ANNUAL", "EARNINGS"]  # SEC 5-class
TYPE_ORDER_FULL = ["10K", "10Q", "8K", "EARNINGS", "CHI"]  # Full corpus 5-class (ANNUAL merged)
TYPE_ORDER_MERGED = ["PERIODIC", "8K", "EARNINGS", "CHI"]  # 4-class (10K+10Q merged)

# -- Phase output directories --
PHASE0_OUTPUT_DIR = Path("output/phase0")
PHASE1_OUTPUT_DIR = Path("output/phase1")
PHASE1_REPR_DIR = PHASE1_OUTPUT_DIR / "representations"
PHASE1_SIM_DIR = PHASE1_OUTPUT_DIR / "similarity_matrices"
PHASE1_CLUSTER_DIR = PHASE1_OUTPUT_DIR / "clustering"
PHASE1_ANALYSIS_DIR = PHASE1_OUTPUT_DIR / "analysis"

# -- Bisection paths --
BISECTION_SIM_PATH = Path("output/phase2/similarity_matrices/S_sem_full.npy")
BISECTION_LABELS_PATH = Path("output/phase2/clustering/full_corpus_labels.npz")
BISECTION_REPR_PATH = Path("output/phase2/representations/full_representations.pkl")
FILING_CSV = Path("output/phase0/sec_filing_types.csv")
BISECTION_OUT_DIR = Path("output/phase2_clustering")

# -- Adaptive LLM paths --
ADAPTIVE_LLM_OUTPUT_DIR = Path("output/phase2_adaptive_llm_cc")

# -- Eigengap paths --
EIGENGAP_OUT_DIR = Path("output/phase2_eigengap_diagnostic")


# -- OT optimal parameters (Phase 1 tuning: alpha=0.5, tau=0.5, eps=0.01 -> Cohen's d=1.0427) --
OT_ALPHA = 0.5
OT_TAU = 0.5
OT_EPSILON = 0.01


def get_processing_dir(doc_id: str, sample_types: Dict[str, str]) -> Path:
    """Return the processing directory corresponding to the document type."""
    return CHI_PROCESSING if sample_types.get(doc_id) == "CHI" else SEC_PROCESSING


def get_embedding_dir(
    doc_id: str,
    sample_types: Dict[str, str],
    provider: str = DEFAULT_EMBED_PROVIDER,
) -> Path:
    """Return the embedding directory corresponding to the document type and provider."""
    if sample_types.get(doc_id) == "CHI":
        return Path(f"datasets/paper/latest/embedding/{provider}/document_embedding")
    return Path(f"datasets/pdfs/latest/embedding/{provider}/document_embedding")
