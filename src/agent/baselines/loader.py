"""Build DocInputs for a (query_idx, doc_id) pair.

Falls back gracefully when a dataset lacks pipeline-processed files:
- No reconstructed.json → text extracted directly from PDF via PyMuPDF.
- No label file → ground_truth returned as "" (a warning is logged).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from agent.rule_runtime.data import (
    extract_ground_truth,
    get_label_filename,
    reconstruct_to_normalized_text,
)
from agent.baselines.base import DocInputs
from agent.baselines.defaults import resolve_dataset_root

logger = logging.getLogger(__name__)


def _pdf_text_fallback(pdf_path: Path) -> str:
    """Extract plain text from a PDF via PyMuPDF when no reconstructed.json exists."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append(page.get_text())
    return "\n\n".join(pages)


def _processing_dir(config: dict[str, Any]) -> Path:
    root = resolve_dataset_root(config)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "processing_mineru"
    return root / "processing"


def _label_dir(config: dict[str, Any]) -> Path:
    root = resolve_dataset_root(config)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "label_mineru"
    return root / "label"


def _pdf_path(config: dict[str, Any], doc_id: str) -> Path:
    """Resolve the original PDF path from the dataset layout."""
    root = resolve_dataset_root(config)
    pdf = root / "raw" / f"{doc_id}.pdf"
    return pdf


def _load_entries(reconstruct_path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    with reconstruct_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("texts", [])
    section_index = {i: e for i, e in enumerate(entries)}
    return entries, section_index


def build_doc_inputs(
    config: dict[str, Any],
    query_idx: int,
    doc_id: str,
) -> DocInputs:
    """Build a DocInputs for one (query_idx, doc_id) pair.

    Args:
        config: Loaded experiment YAML config dict.
        query_idx: Zero-based query index.
        doc_id: Document identifier (filename stem).

    Returns:
        Fully populated DocInputs.
    """
    proc_dir = _processing_dir(config)
    label_dir = _label_dir(config)
    truncate_before = config.get("truncate_before")

    pdf = _pdf_path(config, doc_id)

    reconstruct_path = proc_dir / f"{doc_id}_reconstructed.json"
    if reconstruct_path.exists():
        normalized_text = reconstruct_to_normalized_text(reconstruct_path, truncate_before=truncate_before)
        entries, section_index = _load_entries(reconstruct_path)
    else:
        normalized_text = _pdf_text_fallback(pdf)
        entries, section_index = [], {}

    # Distinguish missing-label-file (expected for unlabeled datasets) from a
    # corrupt label / lookup bug (a real error worth surfacing). When the label
    # directory just doesn't exist we silently skip; for anything else we log.
    ground_truth = ""
    if label_dir.exists():
        try:
            label_filename = get_label_filename({"dataset": config.get("dataset", "pdfs")}, query_idx)
            label_path = label_dir / label_filename
            if label_path.exists():
                ground_truth = extract_ground_truth(label_path, doc_id, query_idx)
        except Exception as exc:
            logger.warning(
                "extract_ground_truth failed for q=%d doc=%s: %s", query_idx, doc_id, exc,
            )

    return DocInputs(
        normalized_text=normalized_text,
        entries=entries,
        section_index=section_index,
        pdf_path=pdf,
        ground_truth=ground_truth,
    )


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return loaded
