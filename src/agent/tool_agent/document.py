"""Document context loading layer — provides minimal full-text context + structured entries for the tool-agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.rule_runtime.data import (
    estimate_tokens,
    get_label_filename,
    reconstruct_to_normalized_text,
)


@dataclass(slots=True)
class DocumentContext:
    """Minimal context for a single document plus its raw structured entries."""

    doc_id: str
    query_idx: int
    normalized_text: str
    ground_truth: str
    # Raw texts[] list from *_reconstructed.json (exposed to agent via structured tools)
    entries: list[dict[str, Any]] = field(default_factory=list)
    # section_id (entries index) → entry, for O(1) lookup by id
    section_index: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.normalized_text)


def _load_ground_truth(
    label_path: Path,
    doc_id: str,
    query_idx: int,
) -> str:
    """Load only the ground_truth value from a label JSON file."""
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data.get("labels", []):
        if entry.get("doc_name") == doc_id and entry.get("question_idx") == query_idx:
            gt = entry.get("ground_truth", "")
            if not gt:
                raise ValueError(f"ground_truth is empty for doc={doc_id} query_idx={query_idx}")
            return gt

    raise ValueError(f"doc_name={doc_id} (query_idx={query_idx}) not found in label file")


def _load_entries(recon_path: Path) -> list[dict[str, Any]]:
    """Read only the texts[] list from a reconstructed JSON file."""
    with recon_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("texts", []))


def _truncate_entries(
    entries: list[dict[str, Any]],
    truncate_before: str | None,
) -> list[dict[str, Any]]:
    """Return entries up to (not including) the first section_header matching truncate_before.

    Matches the semantics of `reconstruct_to_normalized_text` so the entire tool-agent
    data chain (normalized_text / entries / TOC / get_section / get_page) operates over
    the same range. If truncate_before is None or nothing matches, returns all entries.
    """
    if not truncate_before:
        return entries
    pattern = re.compile(truncate_before, re.IGNORECASE)
    for i, entry in enumerate(entries):
        if entry.get("label") != "section_header":
            continue
        text = (entry.get("text") or "").strip()
        if pattern.search(text):
            return entries[:i]
    return entries


def load_document_context(
    doc_id: str,
    query_idx: int,
    processing_dir: Path,
    label_dir: Path,
    truncate_before: str | None = None,
    dataset_name: str = "pdfs",
) -> DocumentContext:
    """Load minimal context and raw structured entries for a single document."""
    recon_path = processing_dir / f"{doc_id}_reconstructed.json"
    normalized_text = reconstruct_to_normalized_text(
        recon_path,
        truncate_before=truncate_before,
    )
    entries_full = _load_entries(recon_path)
    entries = _truncate_entries(entries_full, truncate_before)
    section_index = {i: entry for i, entry in enumerate(entries)}

    label_path = label_dir / get_label_filename({"dataset": dataset_name}, query_idx)
    ground_truth = _load_ground_truth(label_path, doc_id, query_idx)

    return DocumentContext(
        doc_id=doc_id,
        query_idx=query_idx,
        normalized_text=normalized_text,
        ground_truth=ground_truth,
        entries=entries,
        section_index=section_index,
    )
