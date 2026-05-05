"""grabtoc tool: Table-of-Contents-based section lookup in structured JSON documents.

Steps
-----
1. Parse the JSON file and find all nodes whose ``type`` is ``section_header``
   to build the Table of Contents (TOC).
2. For each header node, accumulate the ``content`` of all *subsequent*
   non-header nodes as its text span, stopping at the next header node.
3. Given one or more keywords, match against the TOC headers (case-insensitive
   substring, then token-overlap ranking) and return the matched header(s)
   with their full text spans.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_TOC_CACHE_DIR = _ROOT / "embeddings" / "officeqa" / "toc"


def _toc_cache_path(doc_path: str) -> Path:
    """Return the TOC cache file path for a given document path."""
    stem = Path(doc_path).stem
    return _TOC_CACHE_DIR / f"{stem}_toc.json"


def _load_toc_cache(cache_file: Path) -> list[TocEntry] | None:
    """Load a cached TOC from disk; return None if absent or unreadable."""
    if cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [
                TocEntry(index=e["index"], header=e["header"], text_span=e["text_span"])
                for e in data
            ]
        except Exception:
            pass
    return None


def _save_toc_cache(cache_file: Path, toc: list[TocEntry]) -> None:
    """Persist a TOC to disk; silently ignore write errors."""
    try:
        _TOC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = [
            {"index": e.index, "header": e.header, "text_span": e.text_span}
            for e in toc
        ]
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TocEntry:
    """One entry in the Table of Contents."""
    index: int            # position of the header element in the document
    header: str           # text content of the header
    text_span: list[str] = field(default_factory=list)  # content of subsequent non-header nodes

    def span_text(self) -> str:
        """Return the full text span as a single string."""
        return "\n\n".join(s for s in self.text_span if s)


# ---------------------------------------------------------------------------
# Core logic (importable without LangChain)
# ---------------------------------------------------------------------------

_HEADER_TYPE = "section_header"
# Node types whose content should be excluded from text spans (navigation noise)
_SKIP_TYPES = {"page_header", "page_footer", "page_number"}


def build_toc(elements: list[dict[str, Any]]) -> list[TocEntry]:
    """Parse a flat list of document elements and build a list of TocEntry objects.

    Each TocEntry accumulates the content of all non-header elements that
    follow its header, up to (but not including) the next header.
    """
    toc: list[TocEntry] = []
    current: TocEntry | None = None

    for i, el in enumerate(elements):
        el_type = el.get("type", "")
        content = el.get("content")

        if el_type == _HEADER_TYPE:
            header_text = str(content).strip() if content is not None else ""
            current = TocEntry(index=i, header=header_text)
            toc.append(current)
        elif current is not None and el_type not in _SKIP_TYPES:
            if content is not None:
                text = str(content).strip()
                if text:
                    current.text_span.append(text)

    return toc


def load_toc_from_file(path: str) -> list[TocEntry]:
    """Load a JSON document file and return its TOC."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)

    # Support both top-level list and {"document": {"elements": [...]}} layout
    if isinstance(data, list):
        elements = data
    elif isinstance(data, dict):
        elements = (
            data.get("document", {}).get("elements")
            or data.get("elements")
            or []
        )
    else:
        elements = []

    return build_toc(elements)


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score(entry: TocEntry, query_tokens: set[str]) -> tuple[int, int]:
    """Return (exact_substring_bonus, token_overlap_count) for ranking."""
    header_lower = entry.header.lower()
    # Check if any keyword phrase occurs as a substring
    exact = 1 if any(qt in header_lower for qt in query_tokens) else 0
    overlap = len(query_tokens & _tokenize(entry.header))
    return (exact, overlap)


def search_toc(
    toc: list[TocEntry],
    keywords: list[str],
    top_k: int = 3,
) -> list[TocEntry]:
    """Return the top-k TOC entries best matching the given keywords.

    Matching strategy (priority order):
    1. Exact case-insensitive substring match of any keyword in the header.
    2. Token-overlap count between keyword tokens and header tokens.
    Entries with score (0, 0) are omitted unless nothing else matches.
    """
    # Normalise: combine all keywords into a single query token set, and keep
    # each keyword phrase for substring matching.
    query_tokens = _tokenize(" ".join(keywords))
    keyword_phrases = [kw.lower().strip() for kw in keywords if kw.strip()]

    scored: list[tuple[tuple[int, int], TocEntry]] = []
    for entry in toc:
        header_lower = entry.header.lower()
        exact = 1 if any(phrase in header_lower for phrase in keyword_phrases) else 0
        overlap = len(query_tokens & _tokenize(entry.header))
        score = (exact, overlap)
        scored.append((score, entry))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top_k; skip entries with score (0,0) unless they're all we have
    results = [e for _, e in scored[:top_k] if _[0] != (0, 0)]
    if not results:
        results = [e for _, e in scored[:top_k]]
    return results


def grabtoc_core(
    path: str,
    keywords: list[str],
    top_k: int = 3,
) -> str:
    """Load the JSON file, build the TOC, search, and return formatted results."""
    fp = Path(path)
    if not fp.is_file():
        return f"(file not found: {path})"

    cache_file = _toc_cache_path(path)
    toc = _load_toc_cache(cache_file)
    if toc is None:
        try:
            toc = load_toc_from_file(path)
        except Exception as exc:
            return f"(error parsing JSON: {exc})"
        _save_toc_cache(cache_file, toc)

    if not toc:
        return "(no section_header nodes found in document)"

    matches = search_toc(toc, keywords, top_k=top_k)
    if not matches:
        return "(no matching headers found)"

    parts: list[str] = []
    for entry in matches:
        span = entry.span_text()
        parts.append(
            f"=== Header [{entry.index}]: {entry.header} ===\n{span}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def grabtoc(
    path: str = "",
    keywords: str = "",
    top_k: int = 3,
) -> str:
    """TOC-based section lookup in a structured JSON document.

    Parses the JSON file at ``path``, identifies all ``section_header`` nodes
    as the Table of Contents, then returns the header(s) most relevant to
    ``keywords`` along with their full text spans.

    Args:
        path:     Path to the ``.json`` document file.
        keywords: Comma-separated search terms (e.g. ``"GDP growth,economy"``).
        top_k:    Maximum number of matching sections to return (default 3).

    Returns:
        Each matched header followed by its text span, separated by ``===``.
    """
    p = (path or "").strip()
    if not p:
        return "(no path provided)"

    kws = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kws:
        return "(no keywords provided)"

    return grabtoc_core(p, kws, top_k=int(top_k))
