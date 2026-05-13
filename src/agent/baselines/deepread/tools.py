"""Tool implementations for the DeepRead locate-then-read ReAct loop.

Retrieve(query, k) -> list[RetrieveHit]
ReadSection(section_id, start, end) -> str (capped at 4000 chars)
"""

from __future__ import annotations

from agent.baselines.deepread.index import ParagraphIndex, RetrieveHit

_READ_MAX_CHARS = 4000


def retrieve(index: ParagraphIndex, query: str, k: int = 5) -> list[RetrieveHit]:
    """TF-IDF retrieval over the paragraph index."""
    return index.retrieve(query, k=k)


def read_section(
    index: ParagraphIndex,
    section_id: int,
    start: int = 0,
    end: int | None = None,
) -> str:
    """Read a section's paragraphs, clamped to section bounds and capped at 4000 chars."""
    return index.read_section(section_id, start=start, end=end, max_chars=_READ_MAX_CHARS)
