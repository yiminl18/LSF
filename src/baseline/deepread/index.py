"""Paragraph index and TF-IDF retriever for the DeepRead baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ParagraphCoord:
    """Stable address for a paragraph inside a section."""

    section_id: int
    in_section_order: int


@dataclass(slots=True)
class Paragraph:
    coord: ParagraphCoord
    text: str
    page_no: int


@dataclass(slots=True)
class SectionMeta:
    section_id: int
    heading: str
    level: int
    page_no: int


@dataclass
class RetrieveHit:
    coord: ParagraphCoord
    snippet: str
    score: float


@dataclass
class ParagraphIndex:
    """In-memory OCR paragraph index with lazy TF-IDF retrieval."""

    paragraphs: list[Paragraph] = field(default_factory=list)
    sections: dict[int, SectionMeta] = field(default_factory=dict)
    _vectorizer: Any = field(default=None, repr=False, compare=False)
    _matrix: Any = field(default=None, repr=False, compare=False)

    def _ensure_tfidf(self) -> None:
        if self._matrix is not None:
            return
        if not self.paragraphs:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [p.text for p in self.paragraphs]
        vec = TfidfVectorizer(
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        self._matrix = vec.fit_transform(texts)
        self._vectorizer = vec

    def retrieve(self, query: str, k: int = 5) -> list[RetrieveHit]:
        """Return up to k paragraph hits with coordinates and snippets."""
        import numpy as np

        self._ensure_tfidf()
        if self._vectorizer is None or self._matrix is None or not self.paragraphs:
            return []
        if not query.strip():
            return []

        q_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ q_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:k]

        hits: list[RetrieveHit] = []
        for idx in top_indices:
            score = float(scores[int(idx)])
            if score <= 0:
                break
            para = self.paragraphs[int(idx)]
            hits.append(
                RetrieveHit(
                    coord=para.coord,
                    snippet=para.text[:240],
                    score=score,
                )
            )
        return hits

    def read_section(
        self,
        section_id: int,
        start: int = 0,
        end: int | None = None,
        max_chars: int = 4000,
    ) -> str:
        """Read paragraphs in one section, clamped to section bounds."""
        paras = [p for p in self.paragraphs if p.coord.section_id == section_id]
        paras.sort(key=lambda p: p.coord.in_section_order)
        if not paras:
            return ""

        start = max(0, start)
        if end is None:
            end = len(paras)
        end = min(end, len(paras))
        if end <= start:
            return ""

        return "\n\n".join(p.text for p in paras[start:end])[:max_chars]

    @classmethod
    def from_ocr_result(cls, ocr_result: dict[str, Any]) -> "ParagraphIndex":
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
                text = str(para_dict.get("text", "")).strip()
                if not text:
                    continue
                idx.paragraphs.append(
                    Paragraph(
                        coord=ParagraphCoord(section_id=sid, in_section_order=order),
                        text=text,
                        page_no=int(para_dict.get("page_no", meta.page_no)),
                    )
                )
        return idx

    def to_dict(self) -> dict[str, Any]:
        sections_out: list[dict[str, Any]] = []
        for sid, meta in sorted(self.sections.items()):
            paras = sorted(
                [p for p in self.paragraphs if p.coord.section_id == sid],
                key=lambda p: p.coord.in_section_order,
            )
            sections_out.append(
                {
                    "section_id": meta.section_id,
                    "heading": meta.heading,
                    "level": meta.level,
                    "page_no": meta.page_no,
                    "paragraphs": [
                        {"text": p.text, "page_no": p.page_no}
                        for p in paras
                    ],
                }
            )
        return {"sections": sections_out}
