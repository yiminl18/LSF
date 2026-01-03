from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HeaderNode:
    """
    Light wrapper around a merged.json text node.
    Only contains the fields we need for feature extraction.
    """

    idx_in_texts: int
    text: str
    text_span: str
    page_no: int
    font_size: float
    is_bold: int

    @property
    def combined_text(self) -> str:
        return f"{self.text} {self.text_span}".strip()


@dataclass(frozen=True)
class DocumentContext:
    total_headers: int
    pos_frac: List[float]
    prefix_pattern_change_count: List[int]
    prefix_pattern_approx_distinct: List[int]
    pattern_key: List[Tuple[float, int, str]]


# ---------------------------
# Header filtering / parsing
# ---------------------------


def iter_section_headers(merged: Dict[str, Any]) -> List[HeaderNode]:
    """
    Filter merged.json["texts"] to only content_layer=="body" and label=="section_header",
    preserving the original order (texts array order).
    """

    texts = merged.get("texts") or []
    out: List[HeaderNode] = []
    for idx, node in enumerate(texts):
        if not isinstance(node, dict):
            continue
        if node.get("content_layer") != "body":
            continue
        if node.get("label") != "section_header":
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
        if isinstance(prov0, dict):
            pn = prov0.get("page_no")
            if isinstance(pn, (int, float)):
                page_no = int(pn)

        size = node.get("size")
        font_size = float(size) if isinstance(size, (int, float)) else 0.0

        bold = node.get("bold")
        is_bold = 1 if bool(bold) else 0

        out.append(
            HeaderNode(
                idx_in_texts=idx,
                text=text,
                text_span=text_span,
                page_no=page_no,
                font_size=font_size,
                is_bold=is_bold,
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


def classify_numbering_type(text: str) -> str:
    """
    Coarse prefix classification for section headers.

    Returns one of:
      none, digit, decimal, alpha, roman, bullet, sec_item

    Priority:
      bullet → sec_item → decimal → digit → alpha → roman → none
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
    # roman can over-trigger on normal words; keep it last among non-none
    if _RE_ROMAN.match(text):
        return "roman"
    return "none"


def _starts_with_number(text: str) -> int:
    return 1 if re.match(r"^\s*\d", text or "") else 0


def _starts_with_letter(text: str) -> int:
    return 1 if re.match(r"^\s*[A-Za-z]", text or "") else 0


def font_size_bucket_1pt_round(font_size: float) -> float:
    # round to nearest 1pt, as float (e.g. 8.001 -> 8.0)
    try:
        return float(round(float(font_size) / 1.0) * 1.0)
    except Exception:
        return 0.0


# ---------------------------
# Document-level context
# ---------------------------


def build_document_context(headers_in_order: Sequence[HeaderNode]) -> DocumentContext:
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
        # prefix excludes self: compute stats based on [0, i)
        prefix_pattern_change_count.append(changes)
        prefix_pattern_approx_distinct.append(len(seen))

        # then update state with current key
        if prev_key is not None and key != prev_key:
            changes += 1
        seen.add(key)
        prev_key = key

    return DocumentContext(
        total_headers=total,
        pos_frac=pos_frac,
        prefix_pattern_change_count=prefix_pattern_change_count,
        prefix_pattern_approx_distinct=prefix_pattern_approx_distinct,
        pattern_key=pattern_key,
    )


# ---------------------------
# Node-level features
# ---------------------------


_NUM_TYPES = ("none", "digit", "decimal", "alpha", "roman", "bullet", "sec_item")


def extract_node_features(
    headers_in_order: Sequence[HeaderNode],
    ctx: DocumentContext,
) -> List[Dict[str, float]]:
    """
    Produce node-level features for each header, aligned with headers_in_order.

    Note: Does NOT include rank_index or sim(q,node).
    """

    feats: List[Dict[str, float]] = []
    for i, h in enumerate(headers_in_order):
        nt = classify_numbering_type(h.text)
        d: Dict[str, float] = {
            "starts_with_number": float(_starts_with_number(h.text)),
            "starts_with_letter": float(_starts_with_letter(h.text)),
            "font_size_bucket_1pt_round": float(font_size_bucket_1pt_round(h.font_size)),
            "is_bold": float(int(h.is_bold)),
            "page_no": float(h.page_no),
            "pos_frac": float(ctx.pos_frac[i]) if i < len(ctx.pos_frac) else 0.0,
            "prefix_pattern_change_count": float(
                ctx.prefix_pattern_change_count[i]
            )
            if i < len(ctx.prefix_pattern_change_count)
            else 0.0,
            "prefix_pattern_approx_distinct": float(
                ctx.prefix_pattern_approx_distinct[i]
            )
            if i < len(ctx.prefix_pattern_approx_distinct)
            else 0.0,
        }

        # numbering_type one-hot
        for t in _NUM_TYPES:
            d[f"numtype_{t}"] = 1.0 if nt == t else 0.0

        feats.append(d)
    return feats


# ---------------------------
# Pair-level features (Similarity model input)
# ---------------------------


def extract_pair_features(
    a: HeaderNode,
    b: HeaderNode,
    a_feats: Dict[str, float],
    b_feats: Dict[str, float],
    sim_node_node: float,
) -> Dict[str, float]:
    """
    Pair-level feature vector for Similarity(a,b).

    sim_node_node must be embedding cosine between a.text and b.text.
    """

    a_bucket = float(a_feats.get("font_size_bucket_1pt_round", 0.0))
    b_bucket = float(b_feats.get("font_size_bucket_1pt_round", 0.0))

    # recover numbering_type as argmax over one-hot (safe; caller can also pass it separately)
    def _infer_numtype(d: Dict[str, float]) -> str:
        best_t = "none"
        best_v = -1.0
        for t in _NUM_TYPES:
            v = float(d.get(f"numtype_{t}", 0.0))
            if v > best_v:
                best_v = v
                best_t = t
        return best_t

    a_nt = _infer_numtype(a_feats)
    b_nt = _infer_numtype(b_feats)

    return {
        "sim_node_node": float(sim_node_node),
        "abs_font_bucket_diff": abs(a_bucket - b_bucket),
        "bold_match": 1.0 if int(a.is_bold) == int(b.is_bold) else 0.0,
        "numbering_type_match": 1.0 if a_nt == b_nt else 0.0,
        "abs_page_diff": float(abs(int(a.page_no) - int(b.page_no))),
        "abs_pos_frac_diff": abs(float(a_feats.get("pos_frac", 0.0)) - float(b_feats.get("pos_frac", 0.0))),
        "abs_prefix_change_diff": abs(
            float(a_feats.get("prefix_pattern_change_count", 0.0))
            - float(b_feats.get("prefix_pattern_change_count", 0.0))
        ),
        "abs_prefix_distinct_diff": abs(
            float(a_feats.get("prefix_pattern_approx_distinct", 0.0))
            - float(b_feats.get("prefix_pattern_approx_distinct", 0.0))
        ),
    }


