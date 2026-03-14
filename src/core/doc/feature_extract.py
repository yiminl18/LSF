"""
Document structure feature extraction.

Extracts structural features from document headers for header classification
and similarity learning.

Main functions:
- iter_section_headers(): filter and extract section headers
- classify_numbering_type(): classify header numbering type
- build_document_context(): build document-level context

Data classes:
- HeaderNode: lightweight wrapper for header nodes
- DocumentContext: document-level statistics

Numbering types:
- none: no numbering
- digit: numeric (e.g. 1, 2, 3)
- decimal: decimal numbering (e.g. 1.1, 1.2.3)
- alpha: alphabetic (e.g. A, B, C)
- roman: Roman numerals (e.g. I, II, III)
- bullet: bullet symbols
- sec_item: SEC-style ITEM/PART headings
"""

from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HeaderNode:
    """
    Lightweight wrapper for text nodes in reconstructed.json.

    Contains all visual fields needed for feature extraction.

    Attributes:
        idx_in_texts: index within the texts array
        text: header text
        text_span: text content under the header
        page_no: page number
        font_size: font size
        is_bold: whether bold (0 or 1)
        font_name: font name
        is_all_cap: whether all-caps (0 or 1)
        starts_num: whether starts with a digit (0 or 1)
        starts_letter: whether starts with a letter (0 or 1)
        is_center: whether centered (0 or 1)
    """

    idx_in_texts: int
    text: str
    text_span: str
    page_no: int
    font_size: float
    is_bold: int
    font_name: str
    is_all_cap: int
    starts_num: int
    starts_letter: int
    is_center: int
    content_visuals: Dict[str, List[float]] = field(
        default_factory=lambda: {"sizes": [], "bold_sizes": []}
    )
    processing_path: str = ""
    structure_level: int = 10
    h1_index_norm: float = 0.0
    sibling_index_norm: float = 0.0
    depth: int = 0
    is_first_child: int = 0
    is_last_child: int = 0
    parent_id: int = -1

    @property
    def combined_text(self) -> str:
        """Consistent with get_combined_text(); uses path_text to provide ancestor context."""
        path = self.processing_path or self.text
        return f"{path} {self.text_span}".strip()


@dataclass(frozen=True)
class DocumentContext:
    """
    Document-level context information.

    Contains document-level statistics used for feature extraction.

    Attributes:
        total_headers: total number of headers
        pos_frac: positional fraction for each header (0-1)
        prefix_pattern_change_count: prefix pattern change count
        prefix_pattern_approx_distinct: approximate number of distinct prefix patterns
        pattern_key: pattern key list (font size bucket, bold, numbering type)
        font_size_rank_map: mapping from font size to normalized rank
    """

    total_headers: int
    pos_frac: List[float]
    prefix_pattern_change_count: List[int]
    prefix_pattern_approx_distinct: List[int]
    pattern_key: List[Tuple[float, int, str]]
    font_size_rank_map: Dict[float, float] = field(default_factory=dict)


# ---------------------------
# Header filtering / parsing
# ---------------------------


def iter_section_headers(merged: Dict[str, Any]) -> List[HeaderNode]:
    """
    Filter reconstructed.json["texts"], retaining structurally relevant nodes.

    Includes: section_header

    Preserves original order (texts array order).

    Args:
        merged: processed JSON dictionary

    Returns:
        list of HeaderNode
    """

    texts = merged.get("texts") or []
    out: List[HeaderNode] = []
    for idx, node in enumerate(texts):
        if not isinstance(node, dict):
            continue
        # reconstructed JSON now only contains section_header (page_header/page_footer/footnote merged into text_span)
        label = node.get("label")
        if label != "section_header":
            continue
        # Filter structural noise from overly long spans
        from core.config import MAX_SPAN_WORDS

        text_span_raw = node.get("text_span", "")
        if len(text_span_raw.split()) >= MAX_SPAN_WORDS:
            continue

        text = node.get("text")
        if not isinstance(text, str):
            # keep it robust; empty string still participates in ordering
            text = "" if text is None else str(text)

        text_span = node.get("text_span")
        if not isinstance(text_span, str):
            text_span = "" if text_span is None else str(text_span)

        prov0 = None
        prov = node.get("prov")
        if isinstance(prov, list) and prov:
            if isinstance(prov[0], dict):
                prov0 = prov[0]
        page_no = 0
        got_page = False
        if isinstance(prov0, dict):
            pn = prov0.get("page_no")
            if isinstance(pn, (int, float)):
                page_no = int(pn)
                got_page = True
        # sci-docs format: page_no at node top level rather than inside prov
        if not got_page:
            pn = node.get("page_no") or node.get("header_page")
            if isinstance(pn, (int, float)):
                page_no = int(pn)

        size = node.get("size")
        font_size = float(size) if isinstance(size, (int, float)) else 0.0
        is_bold = 1 if node.get("bold") else 0
        font_name = node.get("font", "")
        is_all_cap = 1 if node.get("all_cap") else 0
        # num_st / letter_st may be missing or all zeros in reconstructed JSON; infer from text
        starts_num = 1 if (text and text[0].isdigit()) else 0
        starts_letter = 1 if (text and text[0].isalpha()) else 0
        is_center = 1 if node.get("is_center") else 0
        content_visuals = node.get("content_visuals") or {"sizes": [], "bold_sizes": []}

        structure = node.get("structure") or {}
        processing_path = structure.get("path_text", "")
        structure_level = structure.get("level_index", 10)
        h1_index_norm = float(structure.get("h1_index_norm", 0.0))
        sibling_index_norm = float(structure.get("sibling_index_norm", 0.0))
        depth = int(structure.get("depth", 0))
        is_first_child = 1 if structure.get("is_first_child") else 0
        is_last_child = 1 if structure.get("is_last_child") else 0
        parent_id = structure.get("parent_id")
        if parent_id is None:
            parent_id = -1
        else:
            parent_id = int(parent_id)

        out.append(
            HeaderNode(
                idx_in_texts=idx,
                text=text,
                text_span=text_span,
                page_no=page_no,
                font_size=font_size,
                is_bold=is_bold,
                font_name=font_name,
                is_all_cap=is_all_cap,
                starts_num=starts_num,
                starts_letter=starts_letter,
                is_center=is_center,
                content_visuals=content_visuals,
                processing_path=processing_path,
                structure_level=structure_level,
                h1_index_norm=h1_index_norm,
                sibling_index_norm=sibling_index_norm,
                depth=depth,
                is_first_child=is_first_child,
                is_last_child=is_last_child,
                parent_id=parent_id,
            )
        )
    return out


# ---------------------------
# numbering_type classification
# ---------------------------


_RE_BULLET = re.compile(r"^\s*(?:[•\-\*]\s+)")
_RE_SEC_ITEM = re.compile(r"^\s*item\s+\d+([a-z])?(\.)?\b", re.IGNORECASE)
_RE_SEC_PART = re.compile(r"^\s*part\s+[ivxlcdm]+\b", re.IGNORECASE)

# e.g. "1.2", "2.3.4", "(1.2)", "1.2)"
_RE_DECIMAL = re.compile(r"^\s*\(?\d+(?:\.\d+)+\)?(?:[.)])?\b")

# e.g. "1", "2.", "3)", "(4)", "10."
_RE_DIGIT = re.compile(r"^\s*\(?\d+\)?(?:[.)])?\b")

# e.g. "A.", "B)", "(a)", "(b)"
_RE_ALPHA = re.compile(r"^\s*\(?[A-Za-z]\)?(?:[.)])\b")

# e.g. "I.", "IV", "(ii)", "(X)"
_RE_ROMAN = re.compile(r"^\s*\(?[IVXLCDMivxlcdm]+\)?(?:[.)])?\b")


@lru_cache(maxsize=200_000)
def classify_numbering_type(text: str) -> str:
    """
    Coarse-grained prefix classification for section headers.

    Determines the numbering type based on the prefix pattern of the header text.

    Args:
        text: header text

    Returns:
        numbering type string, one of:
        none, digit, decimal, alpha, roman, bullet, sec_item

    Priority:
        bullet -> sec_item -> decimal -> digit -> alpha -> roman -> none
    """

    if not text:
        return "none"

    if _RE_BULLET.match(text):
        return "bullet"
    if _RE_SEC_ITEM.match(text) or _RE_SEC_PART.match(text):
        return "sec_item"
    if _RE_DECIMAL.match(text):
        return "decimal"
    if _RE_DIGIT.match(text):
        return "digit"
    if _RE_ALPHA.match(text):
        return "alpha"
    # Roman numerals easily false-match common words; check last
    if _RE_ROMAN.match(text):
        return "roman"
    return "none"


def font_size_bucket_1pt_round(font_size: float) -> float:
    """Round font size to the nearest 1pt. e.g. 8.001 -> 8.0"""
    try:
        return float(round(font_size))
    except (TypeError, ValueError):
        return 0.0


def get_document_font_size_ranks(headers: Sequence[HeaderNode]) -> Dict[float, float]:
    """
    Compute font size rank mapping for all headers in a document.

    Args:
        headers: sequence of header nodes

    Returns:
        {font_size_bucket: normalized_rank} dict, rank range [0, 1], 0 = largest
    """
    # Collect all valid font sizes (bucketed, deduplicated)
    sizes = set()
    for h in headers:
        if h.font_size is not None and h.font_size > 0:
            bucket = font_size_bucket_1pt_round(h.font_size)
            sizes.add(bucket)

    if not sizes:
        return {}

    # Sort descending (largest first)
    sorted_sizes = sorted(sizes, reverse=True)
    n = len(sorted_sizes)

    if n == 1:
        return {sorted_sizes[0]: 0.0}

    # Compute normalized rank: largest=0, smallest=1
    return {size: i / (n - 1) for i, size in enumerate(sorted_sizes)}


# ---------------------------
# Document-level context
# ---------------------------


def build_document_context(headers_in_order: Sequence[HeaderNode]) -> DocumentContext:
    """
    Build document-level context information.

    Iterates over all headers to compute positional fractions, pattern keys,
    prefix pattern change counts, and other statistics.

    Args:
        headers_in_order: ordered sequence of header nodes

    Returns:
        DocumentContext object
    """
    total = len(headers_in_order)
    denom = float(total) if total > 0 else 1.0

    pos_frac: List[float] = []
    pattern_key: List[Tuple[float, int, str]] = []
    for i, h in enumerate(headers_in_order):
        pos_frac.append(float(i) / denom)
        nt = classify_numbering_type(h.text)
        fsb = font_size_bucket_1pt_round(h.font_size)
        pattern_key.append((fsb, int(h.is_bold), nt))

    prefix_pattern_change_count: List[int] = []
    prefix_pattern_approx_distinct: List[int] = []

    seen: set = set()
    changes = 0
    prev_key: Optional[Tuple[float, int, str]] = None

    for i, key in enumerate(pattern_key):
        # Prefix statistics exclude self: computed over [0, i)
        prefix_pattern_change_count.append(changes)
        prefix_pattern_approx_distinct.append(len(seen))

        # Then update state with the current key
        if prev_key is not None and key != prev_key:
            changes += 1
        seen.add(key)
        prev_key = key

    # Compute font size rank mapping
    font_size_rank_map = get_document_font_size_ranks(headers_in_order)

    return DocumentContext(
        total_headers=total,
        pos_frac=pos_frac,
        prefix_pattern_change_count=prefix_pattern_change_count,
        prefix_pattern_approx_distinct=prefix_pattern_approx_distinct,
        pattern_key=pattern_key,
        font_size_rank_map=font_size_rank_map,
    )
