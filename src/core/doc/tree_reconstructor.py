#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import sys
import random
import collections
import math
import hashlib
import sqlite3
import threading
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.doc.font_utils import clean_font_name, is_bold_font
from core.doc.feature_extract import classify_numbering_type
from core.llm.model import llm_call
from core.config import JUDGE_CACHE_DIR

from rapidfuzz.distance import Levenshtein as _rf_levenshtein

# Auxiliary text types (will be merged into the text_span of the nearest section_header)
AUXILIARY_LABELS = {"page_header", "page_footer", "footnote"}
LONG_ALPHA_RUN_RE = re.compile(r"[A-Za-z]{20,}")
DOC_LSF_TEXT_MAX_DIST_RATIO = 0.10
DOC_LSF_TEXT_MIN_DIST_CUTOFF = 2


def normalize_text(text: str) -> str:
    """Normalize text: keep only letters for matching."""
    return re.sub(r"[^a-zA-Z]", "", text).lower()


def levenshtein_distance(s1: str, s2: str, score_cutoff: Optional[int] = None) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if score_cutoff is not None:
        return _rf_levenshtein.distance(s1, s2, score_cutoff=score_cutoff)
    return _rf_levenshtein.distance(s1, s2)


def _normalize_letters_only(text: str) -> str:
    """Keep only letters for text consistency comparison."""
    return re.sub(r"[^a-zA-Z]", "", text or "").lower()


def _normalize_alnum_only(text: str) -> str:
    """Keep letters and digits, lowercased, for word-split detection."""
    return re.sub(r"[^a-zA-Z0-9]", "", text or "").lower()


def _is_spatial_text_divergent(lsf_text: str, block_text: str) -> bool:
    """Detect whether a spatial fallback picked up words from the wrong region (when bbox is wrong)."""
    if not lsf_text or not block_text:
        return False
    lsf_norm = _normalize_letters_only(lsf_text)
    block_norm = _normalize_letters_only(block_text)
    if not lsf_norm or not block_norm:
        return False
    max_len = max(len(lsf_norm), len(block_norm))
    threshold = max(3, int(max_len * 0.5))
    dist = levenshtein_distance(lsf_norm, block_norm, score_cutoff=threshold)
    return dist > threshold


def _convert_docling_bbox_to_lsf_topleft(
    docling_bbox: Dict[str, Any],
    page_height: float,
) -> Tuple[float, float, float, float]:
    """Convert Docling bbox to the TOPLEFT coordinate system used by LSF."""
    left = float(docling_bbox["l"])
    right = float(docling_bbox["r"])
    top = float(docling_bbox["t"])
    bottom = float(docling_bbox["b"])
    coord_origin = str(docling_bbox.get("coord_origin", "BOTTOMLEFT")).upper()

    if coord_origin == "BOTTOMLEFT":
        top, bottom = page_height - top, page_height - bottom
        if top > bottom:
            top, bottom = bottom, top

    return left, right, top, bottom


def _should_prefer_docling_text(lsf_text: str, docling_text: str) -> bool:
    """Prefer Docling text when a suspected word-split error is detected and content is semantically equivalent."""
    if not lsf_text or not docling_text:
        return False

    # Word-split detection: identical after removing spaces → same content, only spacing differs
    lsf_nospace = _normalize_alnum_only(lsf_text)
    doc_nospace = _normalize_alnum_only(docling_text)
    if lsf_nospace and doc_nospace and lsf_nospace == doc_nospace:
        if lsf_text.count(" ") > docling_text.count(" "):
            return True

    if not LONG_ALPHA_RUN_RE.search(lsf_text):
        return False

    lsf_norm = _normalize_letters_only(lsf_text)
    docling_norm = _normalize_letters_only(docling_text)
    if not lsf_norm or not docling_norm:
        return False

    max_len = max(len(lsf_norm), len(docling_norm))
    cutoff = max(
        DOC_LSF_TEXT_MIN_DIST_CUTOFF, int(max_len * DOC_LSF_TEXT_MAX_DIST_RATIO)
    )
    dist = levenshtein_distance(lsf_norm, docling_norm, score_cutoff=cutoff)
    return dist <= cutoff


def _match_lsf_words_inner(
    docling_normalized: str,
    docling_left: float,
    docling_right: float,
    docling_top: float,
    docling_bottom: float,
    tolerance: float,
    page_lsf_words: List[Dict[str, Any]],
    norm_phrases: List[str],
    word_norm_lens: List[int],
    word_bboxes: List[list],
) -> Optional[List[Dict[str, Any]]]:
    """Core matching loop: find a matching LSF word sequence in the precomputed page-level data."""
    docling_norm_len = len(docling_normalized)
    if docling_norm_len == 0:
        return None
    max_edit_distance = min(4, max(2, docling_norm_len // 15))
    right_bound = docling_right + tolerance
    left_bound = docling_left - tolerance
    top_bound = docling_top - tolerance * 2
    bottom_bound = docling_bottom + tolerance * 2
    n_words = len(page_lsf_words)
    first_char = docling_normalized[0]

    anchor_starts: List[int] = []
    other_starts: List[int] = []
    for i in range(n_words):
        bbox_i = word_bboxes[i]
        if bbox_i[0] < left_bound or bbox_i[2] > right_bound:
            continue
        if bbox_i[3] < top_bound or bbox_i[1] > bottom_bound:
            continue
        if word_norm_lens[i] == 0 or norm_phrases[i][0] == first_char:
            anchor_starts.append(i)
        else:
            other_starts.append(i)

    best_match_range: Optional[Tuple[int, int]] = None
    best_score = float("inf")

    for pass_starts in (anchor_starts, other_starts):
        if (
            pass_starts is other_starts
            and best_match_range is not None
            and best_score < 200
        ):
            break
        for start_idx in pass_starts:
            cum_len = 0
            min_left, max_right_val = float("inf"), float("-inf")
            min_top, max_bottom_val = float("inf"), float("-inf")
            for lsf_idx in range(start_idx, n_words):
                cum_len += word_norm_lens[lsf_idx]
                bbox_i = word_bboxes[lsf_idx]
                min_left = min(min_left, bbox_i[0])
                max_right_val = max(max_right_val, bbox_i[2])
                min_top = min(min_top, bbox_i[1])
                max_bottom_val = max(max_bottom_val, bbox_i[3])
                if cum_len < docling_norm_len - max_edit_distance:
                    if bbox_i[2] > right_bound:
                        break
                    continue
                if cum_len > docling_norm_len + max_edit_distance:
                    break
                left_error, right_error = (
                    abs(min_left - docling_left),
                    abs(max_right_val - docling_right),
                )
                top_error, bottom_error = (
                    abs(min_top - docling_top),
                    abs(max_bottom_val - docling_bottom),
                )
                if left_error > tolerance or right_error > tolerance:
                    if bbox_i[2] > right_bound:
                        break
                    continue
                if top_error > tolerance * 2 or bottom_error > tolerance * 2:
                    if bbox_i[2] > right_bound:
                        break
                    continue
                lsf_text_so_far = "".join(norm_phrases[start_idx : lsf_idx + 1])
                if lsf_text_so_far == docling_normalized:
                    return page_lsf_words[start_idx : lsf_idx + 1]
                elif lsf_text_so_far.startswith(docling_normalized):
                    score = left_error + right_error + top_error + bottom_error
                    if score < best_score:
                        best_score, best_match_range = score, (start_idx, lsf_idx)
                else:
                    if abs(cum_len - docling_norm_len) <= max_edit_distance:
                        distance = levenshtein_distance(
                            docling_normalized,
                            lsf_text_so_far,
                            score_cutoff=max_edit_distance,
                        )
                        if distance <= max_edit_distance:
                            score = (
                                distance * 100
                                + left_error
                                + right_error
                                + top_error
                                + bottom_error
                            )
                            if score < best_score:
                                best_score, best_match_range = (
                                    score,
                                    (start_idx, lsf_idx),
                                )
                if bbox_i[2] > right_bound:
                    break
    if best_match_range is not None:
        return page_lsf_words[best_match_range[0] : best_match_range[1] + 1]
    return None


def _prepare_page_match_data(
    page_indexed_words: List[Tuple[int, Dict[str, Any]]], excluded_indices: Set[int]
) -> Optional[Tuple[List[Dict[str, Any]], List[str], List[int], List[list]]]:
    """Precompute page-level matching data."""
    page_lsf_words = [
        word for idx, word in page_indexed_words if idx not in excluded_indices
    ]
    if not page_lsf_words:
        return None
    for word in page_lsf_words:
        if "normalized_phrase" not in word:
            word["normalized_phrase"] = normalize_text(word.get("phrase", ""))
    norm_phrases = [w["normalized_phrase"] for w in page_lsf_words]
    word_norm_lens = [len(p) for p in norm_phrases]
    word_bboxes = [w["bbox"] for w in page_lsf_words]
    return page_lsf_words, norm_phrases, word_norm_lens, word_bboxes


def find_matching_lsf_words(
    docling_text: str,
    docling_bbox: Dict[str, Any],
    page_height: float,
    docling_page: int,
    lsf_words: List[Dict[str, Any]],
    tolerance: float = 10.0,
    excluded_indices: Optional[Set[int]] = None,
    page_indexed_words: Optional[List[Tuple[int, Dict[str, Any]]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Find matching LSF word sequences."""
    lsf_page = docling_page - 1
    if excluded_indices is None:
        excluded_indices = set()
    if page_indexed_words is None:
        page_indexed_words = [
            (idx, word)
            for idx, word in enumerate(lsf_words)
            if word["page"] == lsf_page
        ]
    prepared = _prepare_page_match_data(page_indexed_words, excluded_indices)
    if prepared is None:
        return None
    doc_left, doc_right, doc_top, doc_bottom = _convert_docling_bbox_to_lsf_topleft(
        docling_bbox, page_height
    )
    return _match_lsf_words_inner(
        normalize_text(docling_text),
        doc_left,
        doc_right,
        doc_top,
        doc_bottom,
        tolerance,
        *prepared,
    )


def identify_auxiliary_lsf_words(
    auxiliary_texts: List[Dict[str, Any]],
    lsf_words: List[Dict[str, Any]],
    default_page_height: float = 792.0,
    tolerance: float = 10.0,
    page_sizes: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """Identify LSF word indices that belong to auxiliary text."""
    excluded_indices: Set[int] = set()
    aux_regions_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for aux_item in auxiliary_texts:
        prov = aux_item.get("prov", [])
        if not prov:
            continue
        aux_bbox = prov[0].get("bbox", {})
        if not aux_bbox:
            continue
        lsf_page = prov[0].get("page_no", 1) - 1
        page_height = (
            page_sizes[lsf_page]
            if page_sizes and lsf_page in page_sizes
            else default_page_height
        )
        region = (
            aux_bbox.get("l", 0) - tolerance,
            aux_bbox.get("r", 0) + tolerance,
            page_height - aux_bbox.get("t", 0) - tolerance,
            page_height - aux_bbox.get("b", 0) + tolerance,
        )
        if lsf_page not in aux_regions_by_page:
            aux_regions_by_page[lsf_page] = []
        aux_regions_by_page[lsf_page].append(region)
    if not aux_regions_by_page:
        return excluded_indices
    for idx, word in enumerate(lsf_words):
        page = word.get("page", 0)
        regions = aux_regions_by_page.get(page)
        if regions is None:
            continue
        lsf_bbox = word.get("bbox", [])
        if len(lsf_bbox) < 4:
            continue
        x0, y0, x1, y1 = lsf_bbox
        for min_x, max_x, min_y, max_y in regions:
            if x0 >= min_x and x1 <= max_x and y0 >= min_y and y1 <= max_y:
                excluded_indices.add(idx)
                break
    return excluded_indices


# LLM Prompt Cache (similar cache mechanism as judge_header)
_HEADER_VALIDATION_CACHE_DIR = Path(JUDGE_CACHE_DIR)
_HEADER_VALIDATION_CACHE_DB = (
    _HEADER_VALIDATION_CACHE_DIR / "header_validation_cache.db"
)
_header_cache_thread_local = threading.local()


def _get_header_cache_db():
    """Get the thread-local cache database connection."""
    if not hasattr(_header_cache_thread_local, "connection"):
        _HEADER_VALIDATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _header_cache_thread_local.connection = sqlite3.connect(
            _HEADER_VALIDATION_CACHE_DB
        )
        with _header_cache_thread_local.connection:
            _header_cache_thread_local.connection.execute("""
                CREATE TABLE IF NOT EXISTS header_validations (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT,
                    model TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    return _header_cache_thread_local.connection


def _get_prompt_cache_key(prompt: str, model: str) -> str:
    """Generate a cache key for the prompt."""
    content = f"{model}|{prompt}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _check_prompt_cache(prompt: str, model: str) -> Optional[str]:
    """Check prompt cache."""
    cache_key = _get_prompt_cache_key(prompt, model)
    conn = _get_header_cache_db()
    cursor = conn.execute(
        "SELECT response FROM header_validations WHERE prompt_hash = ?", (cache_key,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None


def _save_prompt_cache(prompt: str, model: str, response: str):
    """Save prompt response to cache."""
    cache_key = _get_prompt_cache_key(prompt, model)
    conn = _get_header_cache_db()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO header_validations (prompt_hash, response, model) VALUES (?, ?, ?)",
            (cache_key, response, model),
        )


# Numbering type → hierarchy weight (higher = higher-level structure)
# Kept in sync with features.py:NUMTYPE_RANK
NUMTYPE_HIERARCHY = {
    "sec_item": 6,  # ITEM 1, PART I — SEC top-level
    "roman": 5,  # I, II, III
    "alpha": 4,  # A, B, C
    "decimal": 3,  # 1.2, 1.2.3 — sub-sections
    "digit": 2,  # 1, 2, 3
    "bullet": 1,  # •, -, *
    "none": 0,
}


class PatternKnowledgeBase:
    def __init__(self, kb_path: Path):
        self.path = kb_path
        self.data = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_decision(
        self, key: str, target_error: float = 0.1, audit_rate: float = 0.05
    ) -> Tuple[Optional[bool], bool]:
        if key not in self.data:
            return None, True

        record = self.data[key]
        pos = record["pos"]
        total = record["total"]

        # 1. Random Audit (Epsilon-Greedy)
        # Randomly check trusted patterns to detect concept drift
        if total > 5 and random.random() < audit_rate:
            return None, True

        # 2. Wilson Score Interval for Statistical Confidence
        # z=1.96 for approx 95% confidence
        z = 1.96
        if total == 0:
            return None, True

        p = pos / total
        denominator = 1 + z**2 / total
        center_adjusted = p + z**2 / (2 * total)
        width = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)

        lower_bound = (center_adjusted - width) / denominator
        upper_bound = (center_adjusted + width) / denominator

        # 3. Decision Logic
        # Trust as Header if Lower Bound is high (e.g., > 0.9)
        if lower_bound > (1.0 - target_error):
            return True, False

        # Trust as Body if Upper Bound is low (e.g., < 0.1)
        if upper_bound < target_error:
            return False, False

        # Ambiguous or Insufficient Data -> Validate
        return None, True

    def update(self, key: str, is_header: bool, model: str = ""):
        if key not in self.data:
            self.data[key] = {"pos": 0, "total": 0}

        self.data[key]["total"] += 1
        if is_header:
            self.data[key]["pos"] += 1

        if model:
            self.data[key]["model"] = model

        self.save()


@dataclass
class Node:
    node_id: int
    text: str
    page: int
    style: Dict[str, Any]  # Expanded to store all visual features
    label: str  # docling label
    children: List["Node"] = field(default_factory=list)
    depth: int = 0
    parent: Optional["Node"] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = {
            "id": self.node_id,
            "text": self.text,
            "page": self.page,
            "style": self.style,
            "label": self.label,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children],
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def _has_raw_header_hint(node: "Node") -> bool:
    """MinerU raw header prior: only applies to explicitly marked nodes."""
    return bool(node.metadata.get("raw_header_hint", False))


def _process_table_block(table: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract structural metadata from a Docling table and generate improved markdown.

    Returns:
        (markdown_text, table_data) or None if no valid cells.
    """
    cells = table.get("data", {}).get("table_cells", [])
    if not cells:
        return None

    sorted_cells = sorted(
        cells,
        key=lambda c: (
            c.get("start_row_offset_idx", 0),
            c.get("start_col_offset_idx", 0),
        ),
    )

    # Extract structural metadata + determine table dimensions
    num_rows = 0
    num_cols = 0
    cell_records: List[Dict[str, Any]] = []
    last_header_row = -1

    for cell in sorted_cells:
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        rs = cell.get("row_span", 1)
        cs = cell.get("col_span", 1)
        is_col_header = bool(cell.get("column_header", False))
        is_row_header = bool(cell.get("row_header", False))
        txt = cell.get("text", "").strip()

        num_rows = max(num_rows, r + rs)
        num_cols = max(num_cols, c + cs)

        if is_col_header:
            last_header_row = max(last_header_row, r)

        cell_records.append(
            {
                "row": r,
                "col": c,
                "text": txt,
                "row_span": rs,
                "col_span": cs,
                "is_column_header": is_col_header,
                "is_row_header": is_row_header,
            }
        )

    table_data = {"num_rows": num_rows, "num_cols": num_cols, "cells": cell_records}

    # Build grid (handle merged cells)
    grid: List[List[str]] = [[""] * num_cols for _ in range(num_rows)]
    for rec in cell_records:
        r, c, txt = rec["row"], rec["col"], rec["text"]
        rs, cs = rec["row_span"], rec["col_span"]
        # Primary cell
        if r < num_rows and c < num_cols:
            grid[r][c] = txt
        # col_span > 1: fill subsequent columns with empty placeholder (maintain column alignment)
        for dc in range(1, cs):
            if c + dc < num_cols and r < num_rows:
                grid[r][c + dc] = ""
        # row_span > 1: leave corresponding positions in subsequent rows empty
        for dr in range(1, rs):
            if r + dr < num_rows and c < num_cols:
                grid[r + dr][c] = ""

    # Generate markdown
    md_lines: List[str] = []
    for row_idx in range(num_rows):
        md_lines.append("| " + " | ".join(grid[row_idx]) + " |")
        # Insert separator after header row
        if row_idx == last_header_row:
            md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
        # No column_header mark: add separator after first row (compatible with old behavior)
        elif last_header_row == -1 and row_idx == 0:
            md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    return "\n".join(md_lines), table_data


def is_point_in_bbox(point, bbox, page_height, tolerance=2.0):
    px, py = point
    # Convert LSF Top-Left to Docling Bottom-Left
    py_bl = page_height - py
    return (bbox[0] - tolerance <= px <= bbox[2] + tolerance) and (
        bbox[3] - tolerance <= py_bl <= bbox[1] + tolerance
    )


def aggregate_words_to_text(words, y_tolerance=3.0):
    """Sort words into rows and then by X position to handle superscripts correctly."""
    if not words:
        return ""
    # Sort by Y center
    sorted_by_y = sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2)

    rows = []
    if sorted_by_y:
        current_row = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            w = sorted_by_y[i]
            prev_w = current_row[-1]
            curr_center_y = (w["bbox"][1] + w["bbox"][3]) / 2
            prev_center_y = (prev_w["bbox"][1] + prev_w["bbox"][3]) / 2

            # If vertical overlap or small gap, treat as same line
            if abs(curr_center_y - prev_center_y) < y_tolerance:
                current_row.append(w)
            else:
                rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))
                current_row = [w]
        rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))

    return " ".join([" ".join([w["phrase"] for w in row]) for row in rows])


def sort_blocks(blocks: List[Dict], page_width: float) -> List[Dict]:
    # Sort by Top (T) descending
    sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1], reverse=True)
    final_order = []
    col_buffer = []

    def flush_buffer():
        if not col_buffer:
            return
        left_col = []
        right_col = []
        for b in col_buffer:
            center_x = (b["bbox"][0] + b["bbox"][2]) / 2
            if center_x < page_width / 2:
                left_col.append(b)
            else:
                right_col.append(b)
        left_col.sort(key=lambda b: b["bbox"][1], reverse=True)
        right_col.sort(key=lambda b: b["bbox"][1], reverse=True)
        final_order.extend(left_col)
        final_order.extend(right_col)
        col_buffer.clear()

    for b in sorted_blocks:
        if col_buffer:
            last_top = col_buffer[-1]["bbox"][1]
            if last_top - b["bbox"][1] > 50.0:
                flush_buffer()

        page_center = page_width / 2
        spine_half_width = page_width * 0.025
        spine_left = page_center - spine_half_width
        spine_right = page_center + spine_half_width

        crosses_spine = (b["bbox"][0] < spine_left) and (b["bbox"][2] > spine_right)
        center_x = (b["bbox"][0] + b["bbox"][2]) / 2
        is_centered = abs(center_x - page_center) < (page_width * 0.1)

        if crosses_spine or is_centered:
            flush_buffer()
            final_order.append(b)
        else:
            col_buffer.append(b)

    flush_buffer()
    return final_order


# Extensible list of grouping strategies (leading keyword + visual features)
_SPLITTING_STRATEGIES: List[str] = [
    "keyword",
    "all_cap",
]  # Can add later: "is_center", "is_underline"


def _extract_leading_keyword(text: str) -> str:
    """Extract the leading word as a grouping key (uppercased, trailing punctuation stripped)."""
    text = text.strip()
    if not text:
        return ""
    first_word = text.split()[0].upper()
    return first_word.rstrip(".:;,")


def _interleave_ratio(group_a: List[int], group_b: List[int]) -> float:
    """Compute the average number of group_b members between each consecutive pair in group_a."""
    if len(group_a) < 2:
        return 0.0
    total = sum(
        sum(1 for d in group_b if lo < d < hi) for lo, hi in zip(group_a, group_a[1:])
    )
    return total / (len(group_a) - 1)


def _generate_splitting(
    members: List[Tuple[int, Node]], strategy: str
) -> Optional[Dict[str, Tuple[List[int], List[Node]]]]:
    """Group same-rank members by the specified strategy.

    Returns:
        {label: (positions, nodes)} or None if the grouping is invalid.
    """
    groups: Dict[str, Tuple[List[int], List[Node]]] = collections.defaultdict(
        lambda: ([], [])
    )
    for flat_idx, node in members:
        if strategy == "keyword":
            label = _extract_leading_keyword(node.text)
        else:
            label = "1" if node.style.get(strategy) else "0"
        groups[label][0].append(flat_idx)
        groups[label][1].append(node)

    if strategy == "keyword":
        if not (2 <= len(groups) <= 5):
            return None
    else:
        if len(groups) != 2:
            return None
        for positions, _ in groups.values():
            if len(positions) < 2:
                return None

    return dict(groups)


def _evaluate_splitting(
    splitting: Dict[str, Tuple[List[int], List[Node]]],
) -> Optional[Tuple[float, Dict[int, float], str]]:
    """Evaluate the hierarchy signal strength of a grouping using the interleave ratio.

    Checks in ascending order of sub-group size whether the sparse group wraps
    the union of all denser groups.

    Returns:
        (signal_strength, {node_id: bonus}, description) or None.
    """
    sorted_groups = sorted(splitting.items(), key=lambda x: len(x[1][0]))
    num_groups = len(sorted_groups)

    bonus_map: Dict[int, float] = {}
    total_signal = 0.0
    desc_parts: List[str] = []

    for i in range(num_groups - 1):
        label_sparse, (sparse_pos, sparse_nodes) = sorted_groups[i]
        dense_pos: List[int] = []
        for j in range(i + 1, num_groups):
            dense_pos.extend(sorted_groups[j][1][0])
        dense_pos.sort()

        ratio_sd = _interleave_ratio(sparse_pos, dense_pos)
        ratio_ds = _interleave_ratio(dense_pos, sparse_pos)

        if ratio_sd > ratio_ds * 2 and ratio_sd >= 1.0:
            bonus = (num_groups - 1 - i) * 5
            for node in sparse_nodes:
                bonus_map[node.node_id] = max(bonus_map.get(node.node_id, 0), bonus)
            total_signal += ratio_sd / max(ratio_ds, 0.01)
            desc_parts.append(
                f"'{label_sparse}' ({len(sparse_pos)}x) wraps rest "
                f"[{ratio_sd:.1f}:{ratio_ds:.1f}] → +{bonus}"
            )

    if not bonus_map:
        return None
    return total_signal, bonus_map, "; ".join(desc_parts)


def _detect_containment_bonus(
    flat_nodes: List[Node],
    is_header_fn,
    rank_fn,
) -> Dict[int, float]:
    """Unified framework: detect sub-level containment patterns among same-rank headers.

    For each same-rank group, tries multiple grouping strategies (leading keyword, all_cap, etc.),
    evaluates hierarchy signal strength of each grouping using the interleave ratio,
    and applies the bonus from the strongest one.
    """
    headers: List[Tuple[int, Node, float]] = []
    for i, n in enumerate(flat_nodes):
        if is_header_fn(n):
            headers.append((i, n, rank_fn(n)))

    if len(headers) < 3:
        return {}

    rank_groups: Dict[float, List[Tuple[int, Node]]] = collections.defaultdict(list)
    for flat_idx, node, rank in headers:
        rank_groups[rank].append((flat_idx, node))

    bonuses: Dict[int, float] = {}

    for rank, members in rank_groups.items():
        if len(members) < 3:
            continue

        best_signal = 0.0
        best_bonuses: Dict[int, float] = {}
        best_desc = ""

        for strategy in _SPLITTING_STRATEGIES:
            splitting = _generate_splitting(members, strategy)
            if splitting is None:
                continue

            result = _evaluate_splitting(splitting)
            if result is None:
                continue

            signal, candidate_bonuses, desc = result
            if signal > best_signal:
                best_signal = signal
                best_bonuses = candidate_bonuses
                best_desc = f"{strategy}: {desc}"

        if best_bonuses:
            for node_id, bonus in best_bonuses.items():
                bonuses[node_id] = max(bonuses.get(node_id, 0), bonus)
            print(f"  [SubLevel] {best_desc}")

    return bonuses


def validate_header_patterns(
    nodes: List[Node],
    body_style: Dict[str, Any],
    kb: PatternKnowledgeBase,
    source_type: str,
    llm_provider: str = "azure",
    *,
    llm_model: str,
) -> Set[Tuple[float, bool, str]]:
    suspicious_counts = collections.Counter()
    style_examples = collections.defaultdict(list)

    body_size = body_style["size"]

    for i, n in enumerate(nodes):
        s_tuple = (n.style["size"], n.style["bold"], n.style["font"])
        if n.label == "section_header" and n.style["size"] <= body_size:
            suspicious_counts[s_tuple] += 1
            if len(style_examples[s_tuple]) < 2:
                context = [c.text for c in nodes[i + 1 : i + 5]]
                style_examples[s_tuple].append((n.text, context))

    validated_styles = set()
    validation_model = f"{llm_provider}:{llm_model}"

    for style, count in suspicious_counts.items():
        if count < 2:
            continue

        kb_key = f"{source_type}_{style[0]}_{style[1]}_{style[2]}_{body_size}"
        decision, needs_validation = kb.get_decision(kb_key, target_error=0.1)

        if not needs_validation:
            if decision:
                print(f"  [KB] Hit! Trusted pattern {style} as Header.")
                validated_styles.add(style)
            else:
                print(f"  [KB] Hit! Trusted pattern {style} as Body.")
            continue

        print(
            f"  [LLM] Verifying uncertain pattern {style} (Count {count}) using {validation_model} ({llm_provider})..."
        )
        examples = style_examples[style]
        is_valid = True

        for text, context in examples:
            ctx_str = "\n".join([f"- {t[:100]}" for t in context])
            prompt = f"Analyze document processing hierarchy. Is '{text}' (Size {style[0]}, Bold {style[1]}, Font '{style[2]}') a structural SECTION HEADER for the text below?\nContext:\n{ctx_str}\nAnswer ONLY YES or NO."

            try:
                cached_response = _check_prompt_cache(prompt, validation_model)
                if cached_response:
                    res = cached_response.strip().upper()
                else:
                    res = (
                        llm_call(
                            prompt,
                            llm_provider=llm_provider,
                            model=llm_model,
                            max_tokens=5,
                        )
                        .strip()
                        .upper()
                    )
                    _save_prompt_cache(prompt, validation_model, res)

                if "YES" not in res:
                    is_valid = False
                    break
            except Exception as e:
                print(f"  [LLM] Error: {e}")
                is_valid = False
                break

        kb.update(kb_key, is_valid, validation_model)

        if is_valid:
            print("  [LLM] ✓ Validated as Header. KB updated.")
            validated_styles.add(style)
        else:
            print("  [LLM] ✗ Validated as Body. KB updated.")

    return validated_styles


def prepare_initial_tree(
    lsf_words_raw: List[Dict],
    docling_data: Dict,
    kb: PatternKnowledgeBase,
    source_type: str,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    parser: str = "docling",
) -> Tuple[Node, Dict[str, Any]]:
    """Build the initial tree without Cross-Encoder semantic correction."""
    # 0. Get Page Sizes (0-indexed)
    page_heights = {}
    page_sizes = {}

    for p_str, m in docling_data.get("pages", {}).items():
        try:
            p_idx = int(p_str) - 1
            h = m.get("size", {}).get("height", 792.0)
            page_heights[p_idx] = h
            page_sizes[p_idx] = m.get("size", {})
        except Exception:
            pass

    # 1. Identify auxiliary text
    auxiliary_items = [
        item
        for item in docling_data.get("texts", [])
        if item.get("label") in AUXILIARY_LABELS
    ]
    excluded_indices = identify_auxiliary_lsf_words(
        auxiliary_items, lsf_words_raw, page_sizes=page_heights
    )

    words_by_page = collections.defaultdict(list)
    for idx, w in enumerate(lsf_words_raw):
        if idx in excluded_indices:
            continue
        page = w.get("page", 0) + 1
        words_by_page[page].append(w)

    # Pre-index all LSF words (by 0-indexed page) for use by find_matching_lsf_words
    lsf_page_indexed: Dict[int, List[Tuple[int, Dict]]] = collections.defaultdict(list)
    for idx, w in enumerate(lsf_words_raw):
        lsf_page_indexed[w.get("page", 0)].append((idx, w))

    spatial_counts = collections.Counter()
    for item in docling_data.get("texts", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        prov = item.get("prov", [{}])[0]
        bbox = prov.get("bbox")
        if not bbox:
            continue
        y_bucket = int(bbox["t"] / 10.0) * 10
        spatial_counts[(text, y_bucket)] += 1

    raw_blocks_by_page = collections.defaultdict(list)
    for item in docling_data.get("texts", []):
        if item.get("label") in AUXILIARY_LABELS:
            continue
        text = item.get("text", "").strip()
        prov = item.get("prov", [{}])[0]
        bbox, page = prov.get("bbox"), prov.get("page_no")
        if not bbox:
            continue
        y_bucket = int(bbox["t"] / 10.0) * 10
        if spatial_counts[(text, y_bucket)] > 8:
            continue
        if page:
            raw_blocks_by_page[page].append(
                {
                    "label": item.get("label", "paragraph"),
                    "page": page,
                    "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                    "text": text,
                    "is_aside_text": item.get("is_aside_text", False),
                }
            )

    for table in docling_data.get("tables", []):
        prov = table.get("prov", [{}])[0]
        page = prov.get("page_no")
        if not page:
            continue
        t_bbox = prov.get("bbox")
        if not t_bbox:
            continue

        result = _process_table_block(table)
        if not result:
            continue
        md_text, table_data = result

        raw_blocks_by_page[page].append(
            {
                "label": "table",
                "page": page,
                "bbox": [t_bbox["l"], t_bbox["t"], t_bbox["r"], t_bbox["b"]],
                "text": md_text,
                "table_data": table_data,
            }
        )

    flat_nodes = []
    all_aside_blocks: list[dict] = []  # Buffer aside_text; append as leaf headers after tree is built
    total_blocks = 0
    levenshtein_match_hits = 0
    spatial_fallback_hits = 0
    parser_text_fallback_hits = 0
    docling_text_replacements = 0
    for page in sorted(raw_blocks_by_page.keys()):
        page_w, page_h = (
            page_sizes.get(page, {}).get("width", 612.0),
            page_sizes.get(page, {}).get("height", 792.0),
        )
        # aside_text is excluded from sort_blocks to avoid rotated sidebar bboxes being inserted after headers
        page_blocks = raw_blocks_by_page[page]
        main_blocks = [b for b in page_blocks if not b.get("is_aside_text")]
        aside_blocks = [b for b in page_blocks if b.get("is_aside_text")]
        all_aside_blocks.extend(aside_blocks)
        sorted_page_blocks = sort_blocks(main_blocks, page_w)
        for b_info in sorted_page_blocks:
            b_bbox = b_info["bbox"]
            block_text = b_info.get("text", "")
            block_words = None
            if b_info["label"] != "table":
                total_blocks += 1

            # Priority: Levenshtein text-sequence match + bbox boundary constraint (skip table text since it's in markdown)
            if block_text and b_info["label"] != "table":
                bbox_dict = {
                    "l": b_bbox[0],
                    "t": b_bbox[1],
                    "r": b_bbox[2],
                    "b": b_bbox[3],
                }
                block_words = find_matching_lsf_words(
                    block_text,
                    bbox_dict,
                    page_h,
                    page,
                    lsf_words_raw,
                    tolerance=10.0,
                    excluded_indices=excluded_indices,
                    page_indexed_words=lsf_page_indexed.get(page - 1, []),
                )
                if block_words:
                    levenshtein_match_hits += 1
            levenshtein_matched = bool(block_words)

            # Fallback: spatial containment match (point-in-bbox)
            if not block_words:
                block_words = [
                    w
                    for w in words_by_page.get(page, [])
                    if is_point_in_bbox(
                        (
                            (w["bbox"][0] + w["bbox"][2]) / 2,
                            (w["bbox"][1] + w["bbox"][3]) / 2,
                        ),
                        b_bbox,
                        page_h,
                    )
                ]
                if block_words and b_info["label"] != "table":
                    spatial_fallback_hits += 1

            # Verify spatial fallback text quality: discard and fall back to parser text if bbox is wrong
            if (
                block_words
                and not levenshtein_matched
                and b_info["label"] != "table"
                and block_text
            ):
                lsf_candidate = aggregate_words_to_text(block_words)
                if _is_spatial_text_divergent(lsf_candidate, block_text):
                    parser_text_fallback_hits += 1
                    block_words = None

            block_meta = {
                "raw_label": b_info["label"],
                # Only MinerU uses the raw header prior to avoid amplifying mis-labeled headers in pdfs/docling.
                "raw_header_hint": parser == "mineru"
                and b_info["label"] == "section_header",
            }
            if b_info.get("table_data"):
                block_meta["table_data"] = b_info["table_data"]

            if not block_words:
                if block_text:
                    flat_nodes.append(
                        Node(
                            node_id=len(flat_nodes),
                            text=block_text,
                            page=page,
                            style={"size": 9.0, "bold": False, "font": "unknown"},
                            label=b_info["label"],
                            metadata=block_meta,
                        )
                    )
                continue

            main_size = collections.Counter(
                [round(w["size"] * 2) / 2 for w in block_words]
            ).most_common(1)[0][0]
            # Fix: prefer font name for bold detection since the bold field in LSF words may be inaccurate.
            # For each word, check both the bold field and the font name.
            bold_values = []
            for w in block_words:
                font_cleaned = clean_font_name(w.get("font", ""))
                bold_from_font = is_bold_font(font_cleaned) if font_cleaned else None
                bold_from_field = bool(w.get("bold", False))
                # Use font name if it provides a clear bold signal; otherwise fall back to the field value.
                # This avoids missing bold cases where the font name doesn't include "bold" but the field is correct.
                final_bold = (
                    bold_from_font if bold_from_font is not None else bold_from_field
                )
                bold_values.append(final_bold)
            is_bold = collections.Counter(bold_values).most_common(1)[0][0]
            main_font = collections.Counter(
                [clean_font_name(w.get("font", "")) for w in block_words]
            ).most_common(1)[0][0]
            # Table uses raw markdown text; non-table defaults to LSF aggregated text
            if b_info["label"] == "table":
                node_text = block_text
            else:
                lsf_text = aggregate_words_to_text(block_words)
                if _should_prefer_docling_text(lsf_text, block_text):
                    node_text = block_text
                    docling_text_replacements += 1
                else:
                    node_text = lsf_text
            is_allcap = int(node_text.isupper()) if b_info["label"] != "table" else 0
            style_dict = {
                "size": main_size,
                "bold": is_bold,
                "font": main_font,
                "all_cap": is_allcap,
                "num_st": 0,
                "is_center": 0,
                "is_underline": 0,
            }
            flat_nodes.append(
                Node(
                    node_id=len(flat_nodes),
                    text=node_text,
                    page=page,
                    style=style_dict,
                    label=b_info["label"],
                    metadata=block_meta,
                )
            )

    if total_blocks > 0:
        print(
            "  [TextAlign] "
            f"levenshtein_match_hits={levenshtein_match_hits}/{total_blocks}, "
            f"spatial_fallback_hits={spatial_fallback_hits}/{total_blocks}, "
            f"parser_text_fallback_hits={parser_text_fallback_hits}, "
            f"docling_text_replacements={docling_text_replacements}"
        )

    if not flat_nodes:
        return Node(node_id=-1, text="Root", page=0, style={}, label="root"), {
            "size": 9.0
        }

    styles_as_tuples = [
        (n.style["size"], n.style["bold"], n.style["font"]) for n in flat_nodes
    ]
    dom = collections.Counter(styles_as_tuples).most_common(1)[0][0]
    body_style = {"size": dom[0], "bold": dom[1], "font": dom[2]}

    validated_header_styles = validate_header_patterns(
        flat_nodes,
        body_style,
        kb,
        source_type,
        llm_provider,
        llm_model=llm_model,
    )

    def get_hierarchical_rank(node: Node) -> float:
        rank = node.style["size"] * 1000
        numtype = classify_numbering_type(node.text)
        rank += NUMTYPE_HIERARCHY.get(numtype, 0) * 10
        if (
            node.style["size"],
            node.style["bold"],
            node.style["font"],
        ) in validated_header_styles:
            rank = max(rank, (body_style["size"] + 0.01) * 1000)
        if _has_raw_header_hint(node):
        # MinerU raw header signal must be at least above body level to stably enter the existing tree pipeline.
            rank = max(rank, (body_style["size"] + 0.01) * 1000)
        rank += 1 if node.style["bold"] else 0
        return rank

    def is_header_node(n: Node) -> bool:
        numtype = classify_numbering_type(n.text)
        return (
            n.style["size"] > body_style["size"]
            or _has_raw_header_hint(n)
            or (n.style["size"], n.style["bold"], n.style["font"])
            in validated_header_styles
            or (n.style["bold"] and n.style["size"] >= body_style["size"])
            or numtype == "sec_item"
        )

    # Occurrence pattern detection: containment relationship among same-rank headers → rank bonus
    containment_bonus = _detect_containment_bonus(
        flat_nodes, is_header_node, get_hierarchical_rank
    )

    def get_boosted_rank(node: Node) -> float:
        return get_hierarchical_rank(node) + containment_bonus.get(node.node_id, 0)

    root = Node(
        node_id=-1,
        text="Document Root",
        page=0,
        style={"size": 999.0, "bold": True, "font": ""},
        label="root",
        depth=0,
    )
    stack = [root]
    for n in flat_nodes:
        if not is_header_node(n):
            n.parent, n.depth = stack[-1], stack[-1].depth + 1
            stack[-1].children.append(n)
            continue
        curr_rank = get_boosted_rank(n)
        while len(stack) > 1 and get_boosted_rank(stack[-1]) <= curr_rank:
            stack.pop()
        n.parent, n.depth = stack[-1], stack[-1].depth + 1
        stack[-1].children.append(n)
        stack.append(n)

    # MinerU: adopt orphan text nodes before the first section_header into that header
    if parser == "mineru" and root.children:
        first_header_idx = None
        for i, child in enumerate(root.children):
            if child.label == "section_header":
                first_header_idx = i
                break
        if first_header_idx is not None and first_header_idx > 0:
            first_header = root.children[first_header_idx]
            if first_header.page == 1:  # 1-indexed
                orphans = root.children[:first_header_idx]
                root.children = root.children[first_header_idx:]
                for orphan in reversed(orphans):
                    orphan.parent = first_header
                    _update_subtree_depth(orphan, first_header.depth + 1)
                    first_header.children.insert(0, orphan)

    # Step two: adopt root-level text nodes on page 1 sandwiched between headers
    # Typical case: author line (12pt italic) between paper title (12pt) and affiliation (12pt)
    # Because font size equals title > body_size, is_header_node misclassifies it as a header,
    # and in stack-based construction it shares rank with the title -> title gets popped -> author becomes root child
    if parser == "mineru" and root.children:
        to_adopt: list[tuple[int, Node]] = []  # (index, target_header)
        last_header: Node | None = None
        for i, child in enumerate(root.children):
            if child.page != 1:
                break
            if child.label == "section_header":
                last_header = child
            elif last_header is not None:
                to_adopt.append((i, last_header))
        # Remove back-to-front to avoid index shifting
        for i, target_header in reversed(to_adopt):
            orphan = root.children.pop(i)
            orphan.parent = target_header
            _update_subtree_depth(orphan, target_header.depth + 1)
            target_header.children.append(orphan)

    # Append aside_text as leaf section_headers at end of root (fully participates in retrieval)
    for block in all_aside_blocks:
        aside_node = Node(
            node_id=len(flat_nodes) + len(all_aside_blocks),
            text=block["text"],
            page=block["page"],
            style={"size": 9.0, "bold": False, "font": "unknown"},
            label="section_header",
            parent=root,
            depth=root.depth + 1,
            metadata={"is_aside_text": True, "raw_label": "text"},
        )
        root.children.append(aside_node)

    return root, body_style


def _update_subtree_depth(node: "Node", new_depth: int) -> None:
    """Recursively update the depth of a node and all its descendants."""
    node.depth = new_depth
    for child in node.children:
        _update_subtree_depth(child, new_depth + 1)


def reconstruct_tree(
    lsf_words_raw: List[Dict],
    docling_data: Dict,
    kb: PatternKnowledgeBase,
    source_type: str,
    llm_provider: str = "azure",
    *,
    llm_model: str,
    parser: str = "docling",
) -> Node:
    """Backward-compatible single-document full reconstruction function."""
    from core.doc.cross_encoder import verify_and_fix_parents

    root, body_style = prepare_initial_tree(
        lsf_words_raw,
        docling_data,
        kb,
        source_type,
        llm_provider,
        llm_model=llm_model,
        parser=parser,
    )
    verify_and_fix_parents(root, body_style)
    return root


def print_node_tree(node: Node, indent=""):
    """Print the tree structure of Node dataclass instances."""
    if node.node_id != -1:
        s = node.style
        style_str = (
            f"({s.get('size')}, {{'B' if s.get('bold') else 'R'}}, {s.get('font')})"
        )
        print(f"{indent}├── [{node.label[:4]}] {style_str} {node.text[:60]}")
    for child in node.children:
        print_node_tree(child, indent + ("    " if node.node_id == -1 else "│   "))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_name", type=str, default="ADOBE_2022_10K")
    parser.add_argument(
        "--dataset", type=str, default="pdfs", help="Dataset name (pdfs/paper)"
    )
    parser.add_argument("--llm-provider", type=str, default="azure")
    parser.add_argument("--model", type=str, required=True, help="LLM model")
    args = parser.parse_args()

    paths = PathManager()
    proc_dir = paths.get_processing_dir(args.dataset)
    lsf_path, docling_path = (
        proc_dir / f"{args.pdf_name}_lsf.json",
        proc_dir / f"{args.pdf_name}_docling.json",
    )

    if not lsf_path.exists() or not docling_path.exists():
        print("Error: Missing files")
        return

    kb_path = paths.get_knowledge_base_path(args.dataset)
    kb = PatternKnowledgeBase(kb_path)
    print(f"Loaded Knowledge Base from: {kb_path} ({len(kb.data)} patterns)")

    with open(lsf_path, "r") as f:
        lsf_words = json.load(f)
    with open(docling_path, "r") as f:
        docling_data = json.load(f)

    print(f"Building Tree for {args.pdf_name}...")
    tree_root = reconstruct_tree(
        lsf_words,
        docling_data,
        kb,
        args.dataset,
        llm_provider=args.llm_provider,
        llm_model=args.model,
    )

    print("\n--- Semantic Tree Preview ---")
    print_node_tree(tree_root)

    output_file = proc_dir / f"{args.pdf_name}_tree.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree_root.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\nTree saved to: {output_file}")


if __name__ == "__main__":
    main()
