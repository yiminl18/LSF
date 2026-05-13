"""Build DocInputs for a (query_idx, doc_id) pair.

Reuses the reconstructed.json loading from agent.rule_runtime.data
and resolves the PDF path from the dataset layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent.rule_runtime.data import (
    extract_ground_truth,
    get_label_filename,
    get_query_text,
    reconstruct_to_normalized_text,
)
from agent.baselines.base import DocInputs


def _processing_dir(config: dict[str, Any]) -> Path:
    root = Path(config["dataset_root"])
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "processing_mineru"
    return root / "processing"


def _label_dir(config: dict[str, Any]) -> Path:
    root = Path(config["dataset_root"])
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "label_mineru"
    return root / "label"


def _pdf_path(config: dict[str, Any], doc_id: str) -> Path:
    """Resolve the original PDF path from the dataset layout."""
    root = Path(config["dataset_root"])
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

    reconstruct_path = proc_dir / f"{doc_id}_reconstructed.json"
    normalized_text = reconstruct_to_normalized_text(reconstruct_path, truncate_before=truncate_before)
    entries, section_index = _load_entries(reconstruct_path)

    label_filename = get_label_filename({"dataset": config.get("dataset", "pdfs")}, query_idx)
    label_path = label_dir / label_filename
    ground_truth = extract_ground_truth(label_path, doc_id, query_idx)

    pdf = _pdf_path(config, doc_id)

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
