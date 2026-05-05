"""Grab tool: keyword-based chunk retrieval from documents.

Input:  text (raw string) or path (UTF-8 file to load) + keyword
Output: paragraphs / table rows containing the keyword (case-insensitive)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Core logic (importable without LangChain)
# ---------------------------------------------------------------------------

def _split_chunks(text: str) -> list[str]:
    """Split text into paragraphs (blank-line separated), keeping non-empty ones."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def grab_passages(text: str, keyword: str) -> list[str]:
    """Return all chunks from ``text`` that contain ``keyword`` (case-insensitive)."""
    kw = keyword.strip().lower()
    if not kw:
        return []
    chunks = _split_chunks(text)
    return [c for c in chunks if kw in c.lower()]


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def grab(text: str = "", keyword: str = "", path: str = "") -> str:
    """Keyword-match paragraphs in a document.

    Provide either ``path`` (a UTF-8 text file to load) or raw ``text``.
    Returns every paragraph that contains ``keyword`` (case-insensitive),
    separated by '---'.  Returns '(no matches found)' when nothing matches.
    """
    p = (path or "").strip()
    if p:
        fp = Path(p)
        if not fp.is_file():
            return f"(file not found: {p})"
        body = fp.read_text(encoding="utf-8", errors="replace")
    else:
        body = text

    kw = (keyword or "").strip()
    if not kw:
        return "(no keyword provided)"

    hits = grab_passages(body, kw)
    if not hits:
        return "(no matches found)"
    return "\n---\n".join(hits)
