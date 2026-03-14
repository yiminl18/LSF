# -*- coding: utf-8 -*-
"""Document conversion: PDF to structured JSON/MD (Docling, LSF, mapping).

Heavy dependencies (docling_tool, lsf_tool) are lazily imported to avoid
triggering docling/fitz at package load time.
"""

from core.doc.pdf_extraction import phrase_visual_pattern_extraction
from core.doc.font_utils import clean_font_name, is_bold_font
from core.doc.feature_extract import (
    HeaderNode,
    DocumentContext,
    iter_section_headers,
    build_document_context,
)


def __getattr__(name: str):
    """Lazily import heavy modules, loading only on explicit access."""
    if name == "docling_to_json":
        from core.doc.docling_tool import to_json as docling_to_json

        return docling_to_json
    if name == "docling_to_md":
        from core.doc.docling_tool import to_md as docling_to_md

        return docling_to_md
    if name == "lsf_to_json":
        from core.doc.lsf_tool import to_json as lsf_to_json

        return lsf_to_json
    raise AttributeError(f"module 'core.doc' has no attribute {name!r}")


__all__ = [
    "docling_to_json",
    "docling_to_md",
    "lsf_to_json",
    "phrase_visual_pattern_extraction",
    "clean_font_name",
    "is_bold_font",
    "HeaderNode",
    "DocumentContext",
    "iter_section_headers",
    "build_document_context",
]
