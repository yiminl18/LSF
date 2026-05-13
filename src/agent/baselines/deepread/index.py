"""ParagraphIndex and TF-IDF retriever for DeepRead.

Stores the structured output of LLMOCR.parse_pdf() and provides
Retrieve(query, k) and ReadSection(section_id, start, end) operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ParagraphCoord:
    """Stable address for a paragraph within the document."""

    section_id: int
    in_section_order: int  # 0-based position within the section


@dataclass(slots=True)
class Paragraph:
    """A single paragraph produced by LLM-OCR."""

    coord: ParagraphCoord
    text: str
    page_no: int


@dataclass(slots=True)
class SectionMeta:
    """Heading-level metadata for a section."""

    section_id: int
    heading: str
    level: int  # 1 for H1, 2 for H2, etc.
    page_no: int


@dataclass
class RetrieveHit:
    coord: ParagraphCoord
    snippet: str
    score: float


@dataclass
class ParagraphIndex:
    """In-memory index of OCR paragraphs with TF-IDF retriever."""

    paragraphs: list[Paragraph] = field(default_factory=list)
    sections: dict[int, SectionMeta] = field(default_factory=dict)
    # Lazy-initialized; populated on first Retrieve call.
    _vectorizer: Any = field(default=None, repr=False, compare=False)
    _matrix: Any = field(default=None, repr=False, compare=False)

    def _ensure_tfidf(self) -> None:
        if self._matrix is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        texts = [p.text for p in self.paragraphs]
        if not texts:
            return
        vec = TfidfVectorizer(
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        mat = vec.fit_transform(texts)
        self._vectorizer = vec
        self._matrix = mat

    def retrieve(self, query: str, k: int = 5) -> list[RetrieveHit]:
        """TF-IDF retrieval; returns up to k hits with coord + snippet."""
        import numpy as np
        self._ensure_tfidf()
        if self._vectorizer is None or self._matrix is None or not self.paragraphs:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ q_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        hits: list[RetrieveHit] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            para = self.paragraphs[int(idx)]
            hits.append(RetrieveHit(
                coord=para.coord,
                snippet=para.text[:200],
                score=float(scores[idx]),
            ))
        return hits

    def read_section(
        self,
        section_id: int,
        start: int = 0,
        end: int | None = None,
        max_chars: int = 4000,
    ) -> str:
        """Concatenate paragraphs in section [start, end) by in_section_order."""
        paras = [
            p for p in self.paragraphs
            if p.coord.section_id == section_id
        ]
        paras.sort(key=lambda p: p.coord.in_section_order)
        if end is None:
            end = len(paras)
        # Clamp to section bounds
        start = max(0, start)
        end = min(end, len(paras))
        selected = paras[start:end]
        text = "\n\n".join(p.text for p in selected)
        return text[:max_chars]

    @classmethod
    def from_ocr_result(cls, ocr_result: dict[str, Any]) -> "ParagraphIndex":
        """Build a ParagraphIndex from the JSON payload produced by LLMOCR."""
        idx = cls()
        for section_dict in ocr_result.get("sections", []):
            sid = int(section_dict["section_id"])
            meta = SectionMeta(
                section_id=sid,
                heading=section_dict.get("heading", ""),
                level=int(section_dict.get("level", 1)),
                page_no=int(section_dict.get("page_no", 0)),
            )
            idx.sections[sid] = meta
            for order, para_dict in enumerate(section_dict.get("paragraphs", [])):
                para = Paragraph(
                    coord=ParagraphCoord(section_id=sid, in_section_order=order),
                    text=para_dict.get("text", ""),
                    page_no=int(para_dict.get("page_no", meta.page_no)),
                )
                idx.paragraphs.append(para)
        return idx

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict (for on-disk cache)."""
        sections_out: list[dict[str, Any]] = []
        for sid, meta in sorted(self.sections.items()):
            paras_in_section = sorted(
                [p for p in self.paragraphs if p.coord.section_id == sid],
                key=lambda p: p.coord.in_section_order,
            )
            sections_out.append({
                "section_id": meta.section_id,
                "heading": meta.heading,
                "level": meta.level,
                "page_no": meta.page_no,
                "paragraphs": [
                    {"text": p.text, "page_no": p.page_no}
                    for p in paras_in_section
                ],
            })
        return {"sections": sections_out}
