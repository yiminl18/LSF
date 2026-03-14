"""core.cluster.common.io -- Shared I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_doc_ids(processing_dir: Path) -> list[str]:
    """Get all doc_ids from *_reconstructed.json files in the directory."""
    return sorted(
        p.stem.replace("_reconstructed", "")
        for p in processing_dir.glob("*_reconstructed.json")
    )


def extract_headers(texts: list[dict]) -> list[tuple[int, dict]]:
    """Extract section_header nodes from the texts array.
    
    Filters the raw document blocks to retain only those labeled as headers, 
    preserving their original index for structural mapping.
    """
    return [
        (i, t)
        for i, t in enumerate(texts)
        if isinstance(t, dict) and t.get("label") == "section_header"
    ]


def classify_doc_type(doc_name: str) -> str:
    """Infer document type from document name using pattern matching.
    
    Categorizes documents into standard financial filing types (10-K, 10-Q, 8-K, 
    etc.) based on conventional filename suffixes.
    """
    name = doc_name.upper()
    if "_10K" in name or "_10-K" in name:
        return "10K"
    if "_10Q" in name or "_10-Q" in name:
        return "10Q"
    if "_8K" in name or "_8-K" in name:
        return "8K"
    if "EARNINGS" in name:
        return "EARNINGS"
    if "ANNUALREPORT" in name or "ANNUAL_REPORT" in name:
        return "ANNUAL"
    return "OTHER"
