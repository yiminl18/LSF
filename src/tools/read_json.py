"""read_json tool: extract text content from a structured JSON document.

The JSON files under ``workload/officeqa/data/json/`` contain document elements
(text, table, figure, section_header, etc.) extracted from PDF files.  This
tool flattens them into a searchable text block so the agent can query them
the same way it queries plain-text files.

Input:  path (JSON file) + optional element_types filter + optional keyword
Output: flat text of all matching elements
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.tools import tool


def _flatten_json_doc(path: str, element_types: list[str] | None = None) -> str:
    """Return a flat text view of all (or filtered) elements in a JSON doc file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    elements = data.get("document", {}).get("elements", [])
    lines: list[str] = []
    for i, el in enumerate(elements):
        etype = el.get("type", "")
        if element_types and etype not in element_types:
            continue
        content = el.get("content")
        if content is None:
            continue
        content_str = str(content).strip()
        if not content_str:
            continue
        lines.append(f"[{etype}] {content_str}")
    return "\n\n".join(lines)


@tool
def read_json(path: str = "", element_types: str = "", keyword: str = "") -> str:
    """Read a structured JSON document file and return its text content.

    ``path`` must be a ``.json`` file (e.g. from ``workload/officeqa/data/json/``).
    ``element_types`` is a comma-separated list to filter by type
       (e.g. ``"table,text,section_header"``); leave empty for all.
    ``keyword`` further filters to elements containing the keyword (case-insensitive).

    Returns the flattened text of matching elements, each prefixed with its type.
    """
    p = (path or "").strip()
    if not p:
        return "(no path provided)"
    fp = Path(p)
    if not fp.is_file():
        return f"(file not found: {p})"

    types = [t.strip() for t in element_types.split(",") if t.strip()] if element_types else None
    try:
        text = _flatten_json_doc(p, types)
    except Exception as exc:
        return f"(error reading JSON: {exc})"

    if keyword:
        kw = keyword.strip().lower()
        paras = [para for para in text.split("\n\n") if kw in para.lower()]
        text = "\n\n".join(paras)

    return text if text.strip() else "(no content matched)"
