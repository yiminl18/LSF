"""TOC builder — constructs a tree-based table of contents from reconstructed.json entries.

Replaces the older normalized_text regex approach. This version:
- Uses entries[].structure.{level, parent_id, path_text} to preserve hierarchy
- Tags each section_header line with `[id=N]` (entries index) so the agent can pass
  the id to `get_section`
- Includes `[Nc]` body character count and a 200-char preview
- Renders H1/H2/H3 with depth-based indentation

`section_id` is the entries array index, stable across turns. Callers look up entries
via DocumentContext.section_index.
"""

from __future__ import annotations

import re
from typing import Any

from agent.tool_agent.document import DocumentContext

_PREVIEW_CHARS = 150
_INDENT_BASE = 2  # spaces per extra depth level
_DEFAULT_MAX_DEPTH = 3  # sections deeper than this get a heading line only (no preview)
_PREVIEW_MIN_BODY_CHARS = 100  # only emit preview when section has substantial content
_RULE_LINE_RE = re.compile(r"[_\-=]{10,}")  # collapse cover-page rules / dividers


def _level_depth(level: str | None) -> int:
    """'H1'→1, 'H2'→2, etc. Other values (Body, None) → 0."""
    if not isinstance(level, str):
        return 0
    if len(level) >= 2 and level[0].upper() == "H" and level[1:].isdigit():
        return int(level[1:])
    return 0


def _compute_body_chars(
    section_id: int, entries: list[dict[str, Any]]
) -> tuple[int, str]:
    """Sum body character counts of all descendants and build a preview from the first body text.

    body_chars: total text characters of all non-header descendants
    preview: first non-empty, non-header descendant text, cleaned and truncated to _PREVIEW_CHARS
    """
    # BFS to collect descendant indices
    descendants: set[int] = set()
    frontier = [section_id]
    while frontier:
        next_frontier: list[int] = []
        for pid in frontier:
            for idx, entry in enumerate(entries):
                if entry.get("structure", {}).get("parent_id") == pid and idx not in descendants:
                    descendants.add(idx)
                    next_frontier.append(idx)
        frontier = next_frontier

    total = 0
    preview = ""
    for idx in sorted(descendants):
        entry = entries[idx]
        if entry.get("label") == "section_header":
            continue
        text = entry.get("text", "") or ""
        total += len(text)
        if not preview and text.strip():
            cleaned = _RULE_LINE_RE.sub("…", text.strip())
            if cleaned.strip("… "):  # skip entries that are purely divider lines
                preview = cleaned[:_PREVIEW_CHARS]
    return total, preview


def _entries_from_normalized_text(normalized_text: str) -> list[dict[str, Any]]:
    """Compatibility fallback for older tests and callers that provide normalized_text only."""
    entries: list[dict[str, Any]] = []
    current_page = 1
    current_section_idx: int | None = None
    body_buffer: list[str] = []

    def flush_body() -> None:
        nonlocal body_buffer
        if current_section_idx is None or not body_buffer:
            body_buffer = []
            return
        text = "\n".join(line for line in body_buffer if line.strip()).strip()
        body_buffer = []
        if not text:
            return
        entries.append(
            {
                "label": "text",
                "text": text,
                "page_no": current_page,
                "structure": {
                    "level": "Body",
                    "parent_id": current_section_idx,
                    "path_text": "",
                },
            }
        )

    for raw_line in (normalized_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        page_match = re.fullmatch(r"\[Page\s+(\d+)\]", line, flags=re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if line.startswith("[Section]"):
            flush_body()
            heading = line.removeprefix("[Section]").strip()
            current_section_idx = len(entries)
            entries.append(
                {
                    "label": "section_header",
                    "text": heading,
                    "page_no": current_page,
                    "structure": {
                        "level": "H1",
                        "parent_id": None,
                        "path_text": heading,
                    },
                }
            )
            continue
        body_buffer.append(line)

    flush_body()
    return entries


def build_toc(
    doc_contexts: list[DocumentContext],
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> str:
    """Build a tree-structured TOC for all documents in the sample.

    Sections deeper than max_depth are listed with a heading line only (no size tag or
    preview) to keep token count bounded.
    """
    blocks = [_format_doc_block(doc, max_depth=max_depth) for doc in doc_contexts]
    return "\n\n".join(blocks).rstrip()


def _format_doc_block(doc: DocumentContext, max_depth: int = _DEFAULT_MAX_DEPTH) -> str:
    entries = getattr(doc, "entries", None)
    if entries is None:
        entries = _entries_from_normalized_text(getattr(doc, "normalized_text", ""))
    if not entries:
        return f"### [{doc.doc_id}] (no structural entries)"

    headers: list[tuple[int, dict[str, Any]]] = [
        (i, e) for i, e in enumerate(entries) if e.get("label") == "section_header"
    ]
    tables_count = sum(1 for e in entries if e.get("label") == "table")
    pages = {e.get("page_no") for e in entries if e.get("page_no") is not None}
    n_pages = len(pages)

    lines: list[str] = [
        f"### [{doc.doc_id}] ({n_pages} pages, {len(headers)} headers, {tables_count} tables)"
    ]

    # Use the minimum non-zero depth as the baseline to avoid large left-padding
    # when a document starts at H2 rather than H1.
    depths = [_level_depth(e.get("structure", {}).get("level")) for _, e in headers]
    base_depth = min((d for d in depths if d > 0), default=1)

    for section_id, entry in headers:
        level = entry.get("structure", {}).get("level") or ""
        depth = _level_depth(level)
        indent = " " * (_INDENT_BASE * max(0, depth - base_depth))
        page = entry.get("page_no")
        page_tag = f"p{page}" if page else "p?"
        heading = (entry.get("text") or "").strip()
        level_tag = level if level else "H?"

        # Deep headers: heading line only, skip BFS size computation to save tokens
        if depth and depth > max_depth:
            lines.append(
                f"{indent}{page_tag}  [id={section_id}] {level_tag}  [Section] {heading}"
            )
            continue

        body_chars, preview = _compute_body_chars(section_id, entries)
        size_tag = f"[{body_chars}c]" if body_chars else "[0c]"

        lines.append(
            f"{indent}{page_tag}  [id={section_id}] {level_tag} {size_tag}  [Section] {heading}"
        )
        # Only emit preview when the section has substantial body content
        if preview and body_chars >= _PREVIEW_MIN_BODY_CHARS:
            lines.append(f"{indent}    | preview: {preview}")

    return "\n".join(lines)
