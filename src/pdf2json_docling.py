#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-file PDF to reconstructed JSON converter."""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import fitz
import orjson
import pdfplumber
import pytesseract
from PIL import Image
from rapidfuzz.distance import Levenshtein as _rf_levenshtein

DEFAULT_OUTPUT_DIR = "result"


def clean_font_name(font_name: Optional[str]) -> str:
    """Strip a PDF subset prefix from a font name."""
    if not font_name:
        return ""
    return re.sub(r"^[A-Z]{6}\+", "", font_name.strip())


def is_bold_font(font_name: Optional[str]) -> bool:
    """Determine whether a font is bold based on its name."""
    if not font_name:
        return False
    font_lower = font_name.lower()
    bold_indicators = ["bold", "heavy", "black", "demibold", "semibold", "extrabold"]
    if any(indicator in font_lower for indicator in bold_indicators):
        return True
    if "-b" in font_lower or "b-" in font_lower:
        return True
    if font_lower.startswith("b") and len(font_lower) > 1 and font_lower[1] in ["-", "_"]:
        return True
    clean = re.sub(r"^[a-z]{6}\+", "", font_lower)
    return bool(re.match(r"cm(bx|mib|ssbx)", clean))


def starts_with_number(text: str) -> bool:
    """Check whether text starts with a digit after leading whitespace."""
    return bool(re.match(r"^\s*\d", text or ""))


def starts_with_letter(text: str) -> bool:
    """Check whether text starts with a letter after leading whitespace."""
    return bool(re.match(r"^\s*[A-Za-z]", text or ""))


_RE_BULLET = re.compile(r"^\s*(?:[\u2022\-\*]\s+)")
_RE_SEC_ITEM = re.compile(r"^\s*item\s+\d+([a-z])?(\.)?\b", re.IGNORECASE)
_RE_SEC_PART = re.compile(r"^\s*part\s+[ivxlcdm]+\b", re.IGNORECASE)
_RE_DECIMAL = re.compile(r"^\s*\(?\d+(?:\.\d+)+\)?(?:[.)])?\b")
_RE_DIGIT = re.compile(r"^\s*\(?\d+\)?(?:[.)])?\b")
_RE_ALPHA = re.compile(r"^\s*\(?[A-Za-z]\)?(?:[.)])\b")
_RE_ROMAN = re.compile(r"^\s*\(?[IVXLCDMivxlcdm]+\)?(?:[.)])?\b")


@lru_cache(maxsize=200_000)
def classify_numbering_type(text: str) -> str:
    """Classify a section-heading prefix into a coarse numbering type."""
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
    if _RE_ROMAN.match(text):
        return "roman"
    return "none"


PDFPLUMBER_WORD_X_TOLERANCE = 1.0


def is_valid_phrase(text: str) -> bool:
    """
    Check whether a text phrase is valid (non-empty, not purely whitespace,
    not purely special characters).

    Args:
        text: The text to validate.

    Returns:
        True if the text is valid, False otherwise.
    """
    if not text:
        return False

    # Step 1: Strip leading/trailing whitespace
    text = text.strip()

    # Step 2: Check if empty after stripping
    if not text:
        return False

    # Step 3: Check if it contains only whitespace characters
    if text.isspace():
        return False

    # Step 4: Check if it contains only special/invisible characters (common in PDFs)
    # Remove common invisible characters and check if meaningful content remains
    cleaned_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    cleaned_text = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned_text)  # zero-width spaces

    if not cleaned_text.strip():
        return False

    # Step 5: Check if the cleaned text has at least one printable character
    return any(c.isprintable() for c in cleaned_text)


def _is_all_caps(text: str) -> bool:
    """
    Check whether the text consists entirely of uppercase letters.

    Args:
        text: The text string.

    Returns:
        True if the text is all uppercase letters.
    """
    if not text:
        return False
    return text.isupper() and text.isalpha()


def _is_centered(
    bbox: Tuple[float, float, float, float], page_width: float, tolerance: float = 0.1
) -> bool:
    """
    Check whether a phrase is horizontally centered on the page.

    Args:
        bbox: Bounding box coordinates (x0, y0, x1, y1).
        page_width: Width of the page.
        tolerance: Centering tolerance as a fraction of page width.

    Returns:
        True if the phrase is centered.
    """
    # Compute phrase center
    phrase_center = (bbox[0] + bbox[2]) / 2
    # Compute page center
    page_center = page_width / 2
    # Compute allowed deviation
    center_tolerance = page_width * tolerance

    return abs(phrase_center - page_center) <= center_tolerance


def _has_underline_word(word: Dict) -> bool:
    """
    Check whether a word has an underline.

    Note: pdfplumber cannot reliably detect underlines.

    Args:
        word: A word dictionary from pdfplumber.

    Returns:
        True if an underline is detected.
    """
    # pdfplumber does not always provide underline information.
    # This is a placeholder -- in practice you may need to analyze raw PDF
    # objects or use other heuristics.
    return False


def extract_phrases_with_pdfplumber(
    pdf_path: str, verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract words from a PDF using pdfplumber.

    Based on the get_patterns method -- extracts individual words along with
    their visual patterns.

    Args:
        pdf_path: Path to the PDF file.
        verbose: Whether to print per-page processing logs.

    Returns:
        A list of dictionaries containing word info and visual patterns.
    """
    # ===== Step 2: Initialize data structures =====
    words_data = []
    word_id = 0

    try:
        # ===== Step 3: Open and iterate over PDF pages =====
        with pdfplumber.open(pdf_path) as pdf:
            if verbose:
                print(f"Processing PDF with {len(pdf.pages)} pages...")

            for page_num, page in enumerate(pdf.pages):
                if verbose:
                    print(f"Processing page {page_num + 1}...")

                # ===== Step 4: Extract words and their attributes =====
                # use_text_flow=True preserves reading order
                # extra_attrs extracts font name and size
                words = page.extract_words(
                    use_text_flow=True,
                    x_tolerance=PDFPLUMBER_WORD_X_TOLERANCE,
                    extra_attrs=["fontname", "size"],
                )

                # ===== Step 5: Process each word =====
                for word in words:
                    # Skip empty words
                    if not word["text"].strip():
                        continue

                    # Create bounding box
                    bbox = (word["x0"], word["top"], word["x1"], word["bottom"])

                    # Get font info (strip subset prefix)
                    font_name = clean_font_name(word.get("fontname", "Unknown"))
                    font_size = round(word.get("size", 12), 3)

                    # ===== Step 6: Compute visual pattern features =====
                    is_bold = is_bold_font(font_name)
                    is_all_cap = _is_all_caps(word["text"])
                    is_num_start = starts_with_number(word["text"])
                    is_letter_start = starts_with_letter(word["text"])
                    is_center = _is_centered(bbox, page.width)
                    is_underline = _has_underline_word(word)

                    # ===== Step 7: Build word dictionary =====
                    word_dict = {
                        "id": word_id,
                        "phrase": word["text"],  # keep 'phrase' key for compatibility
                        "bbox": bbox,
                        "page": page_num,
                        "font": font_name,
                        "size": font_size,
                        "bold": 1 if is_bold else 0,
                        "all_cap": 1 if is_all_cap else 0,
                        "num_st": 1 if is_num_start else 0,
                        "letter_st": 1 if is_letter_start else 0,
                        "is_center": 1 if is_center else 0,
                        "is_underline": 1 if is_underline else 0,
                    }

                    words_data.append(word_dict)
                    word_id += 1

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

    return words_data


def phrase_visual_pattern_extraction(
    file_path: str, verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract phrases and their visual patterns from a PDF file.

    Automatically detects the PDF type (text-layer or scanned) and selects
    the appropriate extraction method. Scanned PDFs use OCR; text-layer PDFs
    use pdfplumber.

    Args:
        file_path: Path to the PDF file.
        verbose: Whether to print processing logs.

    Returns:
        A list of phrase objects, each containing:
        - phrase: Extracted text.
        - bbox: Bounding box (x0, y0, x1, y1).
        - page: Page number (0-indexed).
        - font: Font name.
        - size: Font size.
        - bold: Whether bold (0 or 1).
        - is_underline: Whether underlined (0 or 1).
        - all_cap: Whether all caps (0 or 1).
        - num_st: Whether starts with a digit (0 or 1).
        - letter_st: Whether starts with a letter (0 or 1).
        - is_center: Whether centered (0 or 1).
    """

    def is_scanned_pdf(doc) -> bool:
        """Determine whether the PDF is scanned (image-based) or text-based."""
        text_len = 0
        num_pages_to_check = min(3, len(doc))
        for idx in range(num_pages_to_check):
            text = doc[idx].get_text()
            text_len += len(text.strip())
        return text_len < 100

    def extract_text_from_scanned_pdf(doc) -> List[Dict[str, Any]]:
        """Extract text from a scanned PDF using OCR."""
        phrases = []
        phrase_id = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page dimensions
            rect = page.rect
            width = rect.width

            # Convert page to image
            mat = fitz.Matrix(2, 2)  # scale factor to improve OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            # Convert to PIL image
            img = Image.open(io.BytesIO(img_data))

            # Extract text with bounding boxes using OCR
            try:
                # Use pytesseract for detailed output
                ocr_data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6",  # assume a uniform block of text
                )

                # Process OCR results
                for i in range(len(ocr_data["text"])):
                    text = ocr_data["text"][i].strip()
                    # Filter out empty phrases, whitespace-only phrases, and special characters
                    if (
                        is_valid_phrase(text) and int(ocr_data["conf"][i]) > 30
                    ):  # confidence threshold
                        # Get bounding box (normalized to page coordinates)
                        x0 = ocr_data["left"][i] / 2  # divide by scale factor
                        y0 = ocr_data["top"][i] / 2
                        x1 = x0 + (ocr_data["width"][i] / 2)
                        y1 = y0 + (ocr_data["height"][i] / 2)

                        # Estimate font attributes from OCR
                        font_size = ocr_data["height"][i] / 2

                        # Simple heuristic for bold detection
                        # High confidence combined with large height may indicate bold
                        is_bold = (
                            1
                            if (int(ocr_data["conf"][i]) > 70 and font_size > 20)
                            else 0
                        )

                        # Simple heuristic for underline detection
                        # OCR cannot reliably detect underlines, so set to 0.
                        # In practice, underlined text may be detected as a separate element.
                        is_underline = 0

                        # Check if all letters are uppercase
                        all_cap = 1 if text.isupper() and text.isalpha() else 0

                        # Check if the phrase starts with a digit
                        num_st = 1 if text and text[0].isdigit() else 0

                        # Check if the phrase starts with a letter
                        letter_st = 1 if text and text[0].isalpha() else 0

                        # Check if the phrase is centered on the page
                        page_width = width
                        phrase_center_x = (x0 + x1) / 2
                        page_center_x = page_width / 2
                        # Heuristic: considered centered if within 20% of page center
                        center_threshold = page_width * 0.2
                        is_center = (
                            1
                            if abs(phrase_center_x - page_center_x) <= center_threshold
                            else 0
                        )

                        phrases.append(
                            {
                                "id": phrase_id,
                                "phrase": text,
                                "bbox": (x0, y0, x1, y1),
                                "page": page_num,
                                "font": "OCR_Detected",
                                "size": font_size,
                                "bold": is_bold,
                                "is_underline": is_underline,
                                "all_cap": all_cap,
                                "num_st": num_st,
                                "letter_st": letter_st,
                                "is_center": is_center,
                            }
                        )
                        phrase_id += 1

            except Exception as e:
                print(f"OCR failed for page {page_num}: {e}")
                continue

        return phrases

    try:
        # Open the PDF document
        doc = fitz.open(file_path)

        if not doc:
            raise ValueError("Could not open PDF file")

        # Determine whether the PDF is scanned or text-layer
        scanned = is_scanned_pdf(doc)

        if scanned:
            if verbose:
                print("Detected scanned PDF - using OCR")
            phrases = extract_text_from_scanned_pdf(doc)
        else:
            if verbose:
                print("Detected normal PDF - using pdfplumber")
            # Use pdfplumber for text-layer PDFs
            phrases = extract_phrases_with_pdfplumber(file_path, verbose=verbose)

        # Clean up resources
        doc.close()

        return phrases

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []


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

    # Word-split detection: identical after removing spaces means same content, only spacing differs.
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


# Numbering type to hierarchy weight (higher = higher-level structure)
# Kept in sync with features.py:NUMTYPE_RANK
NUMTYPE_HIERARCHY = {
    "sec_item": 6,  # ITEM 1, PART I - SEC top-level
    "roman": 5,  # I, II, III
    "alpha": 4,  # A, B, C
    "decimal": 3,  # 1.2, 1.2.3 - sub-sections
    "digit": 2,  # 1, 2, 3
    "bullet": 1,  # bullet, -, *
    "none": 0,
}


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


def _score_splitting(
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
                f"[{ratio_sd:.1f}:{ratio_ds:.1f}] -> +{bonus}"
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
    scores hierarchy signal strength of each grouping using the interleave ratio,
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

            result = _score_splitting(splitting)
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
    source_type: str = "",
) -> Set[Tuple[float, bool, str]]:
    """Return styles promoted as headers by deterministic visual rules.

    Ambiguous body-sized parser headers are left as body text unless other
    visual/numbering signals classify the individual node as a header.
    """
    suspicious_counts = collections.Counter()
    body_size = body_style["size"]

    for n in nodes:
        s_tuple = (n.style["size"], n.style["bold"], n.style["font"])
        if n.label == "section_header" and n.style["size"] <= body_size:
            suspicious_counts[s_tuple] += 1

    skipped = sum(count for count in suspicious_counts.values() if count >= 2)
    if skipped:
        print(
            "  [HeaderValidation] "
            f"skipped {skipped} ambiguous body-sized parser headers"
        )

    return set()


def prepare_initial_tree(
    lsf_words_raw: List[Dict],
    docling_data: Dict,
    source_type: str = "",
) -> Tuple[Node, Dict[str, Any]]:
    """Build the initial structural tree."""
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

            # Priority: Levenshtein text-sequence match + bbox bounds; skip table markdown.
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
        source_type,
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
        rank += 1 if node.style["bold"] else 0
        return rank

    def is_header_node(n: Node) -> bool:
        numtype = classify_numbering_type(n.text)
        return (
            n.style["size"] > body_style["size"]
            or (n.style["size"], n.style["bold"], n.style["font"])
            in validated_header_styles
            or (n.style["bold"] and n.style["size"] >= body_style["size"])
            or numtype == "sec_item"
        )

    # Occurrence pattern detection: containment relationship among same-rank headers adds a rank bonus.
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

    # Append aside_text as leaf section headers so sidebar text remains addressable.
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
    source_type: str = "",
) -> Node:
    """Backward-compatible single-document reconstruction function."""
    root, _body_style = prepare_initial_tree(
        lsf_words_raw,
        docling_data,
        source_type,
    )
    return root


def print_node_tree(node: Node, indent=""):
    """Print the tree structure of Node dataclass instances."""
    if node.node_id != -1:
        s = node.style
        style_str = (
            f"({s.get('size')}, {{'B' if s.get('bold') else 'R'}}, {s.get('font')})"
        )
        print(f"{indent}|-- [{node.label[:4]}] {style_str} {node.text[:60]}")
    for child in node.children:
        print_node_tree(child, indent + ("    " if node.node_id == -1 else "|   "))


@lru_cache(maxsize=1)
def _get_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    except Exception as exc:
        raise SystemExit(
            "Docling is not installed or cannot be imported. "
            "Install it in the active environment before running conversion.\n"
            f"Error: {exc}"
        ) from exc
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _docling_to_json_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Path:
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    conv_res = _get_docling_converter().convert(input_file)
    doc_dict = conv_res.document.export_to_dict()

    if "texts" in doc_dict:
        def get_sort_key(text_item):
            prov = text_item.get("prov", [])
            if not prov:
                return (0, 0)
            prov_item = prov[0]
            page_no = prov_item.get("page_no", 0)
            bbox = prov_item.get("bbox", {})
            return (page_no, -bbox.get("t", 0))

        doc_dict["texts"].sort(key=get_sort_key)

    if output_path is None:
        result_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        output_file = result_dir / f"{input_file.stem}_docling.json"
    else:
        output_file = Path(output_path).expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def _lsf_to_json_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Path:
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    data = phrase_visual_pattern_extraction(str(input_file), verbose=verbose)
    if output_path is None:
        result_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        output_file = result_dir / f"{input_file.stem}_lsf.json"
    else:
        output_file = Path(output_path).expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def _should_preserve_leaf_header(node: Node) -> bool:
    return bool(node.metadata.get("is_aside_text", False))


def flatten_tree_to_items(root: Node) -> list[dict]:
    """Flatten the reconstructed tree into the canonical `texts` array."""
    items: list[dict] = []
    total_h1 = sum(1 for c in root.children if c.style.get("size", 0) >= 14.0)

    stack: list[tuple[Node, int, int, int]] = []
    for i in range(len(root.children) - 1, -1, -1):
        stack.append((root.children[i], -1, i, 1))

    current_h1_count = 0
    while stack:
        node, parent_id, sibling_idx, depth = stack.pop()

        if depth == 1:
            current_h1_count += 1

        total_siblings = (
            len(node.parent.children) if node.parent else len(root.children)
        )

        structure = {
            "level": "Body",
            "level_index": depth,
            "parent_id": parent_id if parent_id != -1 else None,
            "path_text": node.text,
            "depth": depth,
            "h1_index_norm": (current_h1_count - 1) / total_h1 if total_h1 else 0,
            "sibling_index_norm": sibling_idx / total_siblings if total_siblings else 0,
            "is_first_child": sibling_idx == 0,
            "is_last_child": sibling_idx == total_siblings - 1,
        }

        is_header_in_tree = len(node.children) > 0
        preserve_leaf_header = _should_preserve_leaf_header(node)
        if is_header_in_tree or preserve_leaf_header:
            structure["level"] = f"H{depth}"

        if is_header_in_tree or preserve_leaf_header:
            final_label = "section_header"
        elif node.label == "section_header":
            final_label = "text"
        else:
            final_label = node.label

        if parent_id >= 0:
            parent_path = items[parent_id]["structure"]["path_text"]
            structure["path_text"] = (
                f"{parent_path} | {node.text}"
                if final_label == "section_header"
                else parent_path
            )
        elif final_label != "section_header":
            structure["path_text"] = ""

        style = node.style
        current_id = len(items)
        item = {
            "text": node.text,
            "text_span": "",
            "size": style.get("size"),
            "bold": int(style.get("bold", 0)),
            "font": style.get("font"),
            "all_cap": style.get("all_cap", 0),
            "num_st": style.get("num_st", 0),
            "is_center": style.get("is_center", 0),
            "is_underline": style.get("is_underline", 0),
            "label": final_label,
            "page_no": node.page,
            "structure": structure,
        }
        if final_label == "section_header":
            item["header_page"] = node.page
        if node.metadata.get("table_data"):
            item["table_data"] = node.metadata["table_data"]

        items.append(item)

        for i in range(len(node.children) - 1, -1, -1):
            stack.append((node.children[i], current_id, i, depth + 1))

    children_map: dict[int, list[int]] = {}
    for idx, item in enumerate(items):
        parent_id = item["structure"].get("parent_id")
        if parent_id is not None:
            children_map.setdefault(parent_id, []).append(idx)

    for idx, item in enumerate(items):
        if item["label"] != "section_header":
            continue
        span_texts = [
            items[child_idx].get("text", "")
            for child_idx in children_map.get(idx, [])
            if items[child_idx]["label"] != "section_header"
            and items[child_idx].get("text")
        ]
        item["text_span"] = " ".join(span_texts)

    return items


def break_node_references(node: Node) -> None:
    """Break parent/child cycles after serialization."""
    for child in node.children:
        break_node_references(child)
    node.parent = None
    node.children = []


def _run_structural_parser(pdf_path: Path, output_dir: Path) -> Path:
    return _docling_to_json_file(pdf_path, output_dir=output_dir)


def _run_lsf_parser(pdf_path: Path, output_dir: Path) -> Path:
    return _lsf_to_json_file(pdf_path, output_dir=output_dir, verbose=False)


def _load_docling_like_data(
    structural_json_path: Path,
) -> dict:
    return orjson.loads(structural_json_path.read_bytes())


def build_reconstructed_data(
    pdf_path: str | Path,
    lsf_words: list[dict],
    docling_data: dict,
) -> dict:
    """Build reconstructed JSON data from already-parsed LSF and structural data."""
    pdf = Path(pdf_path).expanduser().resolve()
    root: Optional[Node] = None
    try:
        root, _body_style = prepare_initial_tree(
            lsf_words,
            docling_data,
            source_type="single_pdf",
        )

        return {
            "doc_name": pdf.stem,
            "origin": docling_data.get("origin") or {"filename": pdf.name},
            "texts": flatten_tree_to_items(root),
        }
    finally:
        if root is not None:
            break_node_references(root)


@contextmanager
def _intermediate_dir(
    output_path: Path,
    work_dir: Optional[str | Path],
    keep_intermediate: bool,
) -> Iterator[Path]:
    if work_dir is not None:
        path = Path(work_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return

    if keep_intermediate:
        path = output_path.with_name(f"{output_path.stem}_intermediate")
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="lsf_reconstruct_") as tmp:
        yield Path(tmp)


def pdf_to_reconstructed_json(
    pdf_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    work_dir: Optional[str | Path] = None,
    keep_intermediate: bool = False,
) -> Path:
    """Convert one PDF into a reconstructed JSON file.

    Args:
        pdf_path: Source PDF path.
        output_path: Destination JSON path. Defaults to
            `<pdf_stem>_reconstructed.json` next to the PDF.
        work_dir: Optional directory for parser/LSF intermediate JSON files.
        keep_intermediate: Keep intermediate files in
            `<output_stem>_intermediate` when `work_dir` is not supplied.

    Returns:
        Path to the reconstructed JSON file.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    if output_path is None:
        out = pdf.with_name(f"{pdf.stem}_reconstructed.json")
    else:
        out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with _intermediate_dir(out, work_dir, keep_intermediate) as intermediate:
        structural_path = _run_structural_parser(pdf, intermediate)
        lsf_path = _run_lsf_parser(pdf, intermediate)

        docling_data = _load_docling_like_data(structural_path)
        lsf_words = orjson.loads(lsf_path.read_bytes())
        reconstructed = build_reconstructed_data(
            pdf,
            lsf_words,
            docling_data,
        )

    out.write_bytes(
        orjson.dumps(
            reconstructed,
            option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
        )
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one PDF to reconstructed JSON"
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <pdf_stem>_reconstructed.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate parser/LSF JSON files",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files next to the output when --work-dir is omitted",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = pdf_to_reconstructed_json(
        args.pdf,
        args.output,
        work_dir=args.work_dir,
        keep_intermediate=args.keep_intermediate,
    )
    print(output_path)
    return 0


__all__ = ["pdf_to_reconstructed_json", "build_reconstructed_data", "flatten_tree_to_items"]


if __name__ == "__main__":
    raise SystemExit(main())
