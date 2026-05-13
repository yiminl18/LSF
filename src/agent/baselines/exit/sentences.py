"""Sentence splitter for EXIT baseline.

Mirrors EXIT's upstream sentence segmentation approach (spaCy senter),
with a pure-regex fallback so the extractor works without spaCy installed.
"""

from __future__ import annotations

import re


_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Tries spaCy's 'senter' pipeline first (matching upstream EXIT).
    Falls back to a simple regex split if spaCy is unavailable.

    Returns a list of non-empty sentence strings preserving original order.
    """
    if not text or not text.strip():
        return []

    try:
        import spacy  # type: ignore[import]
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )
        nlp.enable_pipe("senter")
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sentences:
            return sentences
    except Exception:
        pass

    # Regex fallback
    parts = _SENTENCE_BOUNDARY.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]
