"""Tool functions used by the DeepRead locate/read loop."""

from __future__ import annotations

from baseline.deepread.index import ParagraphIndex, RetrieveHit

_READ_MAX_CHARS = 4000


def retrieve(index: ParagraphIndex, query: str, k: int = 5) -> list[RetrieveHit]:
    return index.retrieve(query, k=k)


def read_section(
    index: ParagraphIndex,
    section_id: int,
    start: int = 0,
    end: int | None = None,
) -> str:
    return index.read_section(section_id, start=start, end=end, max_chars=_READ_MAX_CHARS)
