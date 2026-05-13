"""Shared interfaces for paper-baseline extractors.

All extractors implement the BaselineExtractor protocol and accept/return
the common DocInputs / ExtractionResult types defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.pipeline.e2e_utils.cache import CachedLLMCaller


@dataclass(frozen=True, slots=True)
class DocInputs:
    """Input bundle built by loader.py for a single (query_idx, doc_id) pair."""

    normalized_text: str
    """Plain-text rendition of the reconstructed document (from data.reconstruct_to_normalized_text)."""

    entries: list[dict[str, Any]]
    """Raw entries from the reconstructed.json 'texts' list."""

    section_index: dict[int, dict[str, Any]]
    """Map from entry index to its entry dict for fast lookup."""

    pdf_path: Path
    """Absolute path to the original PDF (for DeepRead OCR and MDocAgent)."""

    ground_truth: str
    """Ground-truth answer string; never passed to the model, only used for scoring."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Output from a baseline extractor for one (query, doc) pair."""

    generated_answer: str
    trace: dict[str, Any]
    """Per-baseline trace (selected chunks, OCR coords, agent transcripts, etc.)."""

    cost_usd: float
    latency_ms: float


class BaselineExtractor(Protocol):
    """Protocol every baseline extractor must satisfy."""

    name: str

    def extract(
        self,
        *,
        query_idx: int,
        query_text: str,
        doc_id: str,
        doc_inputs: DocInputs,
        cached_caller: CachedLLMCaller,
    ) -> ExtractionResult: ...
