"""
PDF phrase extraction and visual pattern recognition.

Extracts phrases and their visual attributes (font, size, bold, centering, etc.)
from PDF documents. Supports both native PDFs and scanned PDFs (via OCR).
"""

import re
import io
from typing import Any, Dict, List, Tuple

import fitz
import pytesseract
from PIL import Image
import pdfplumber

from core.utils.text import starts_with_number, starts_with_letter
from core.doc.font_utils import clean_font_name, is_bold_font

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

    Automatically detects the PDF type (native or scanned) and selects
    the appropriate extraction method. Scanned PDFs use OCR; native PDFs
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

        # Determine whether the PDF is scanned or native
        scanned = is_scanned_pdf(doc)

        if scanned:
            if verbose:
                print("Detected scanned PDF - using OCR")
            phrases = extract_text_from_scanned_pdf(doc)
        else:
            if verbose:
                print("Detected normal PDF - using pdfplumber")
            # Use pdfplumber for native PDFs
            phrases = extract_phrases_with_pdfplumber(file_path, verbose=verbose)

        # Clean up resources
        doc.close()

        return phrases

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []


def calculate_bbox_overlap(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Compute the overlap between two bounding boxes.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1).
        bbox2: Second bounding box (x0, y0, x1, y1).

    Returns:
        The IoU (Intersection over Union) value of the overlap region.
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Compute bounds of the intersection region
    x0_intersect = max(x0_1, x0_2)
    y0_intersect = max(y0_1, y0_2)
    x1_intersect = min(x1_1, x1_2)
    y1_intersect = min(y1_1, y1_2)

    # Check if an intersection exists
    if x0_intersect >= x1_intersect or y0_intersect >= y1_intersect:
        return 0.0

    # Compute intersection area
    intersection_area = (x1_intersect - x0_intersect) * (y1_intersect - y0_intersect)

    # Compute union area
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area

    # Return IoU
    return intersection_area / union_area if union_area > 0 else 0.0


def find_matching_phrase(
    section_text: str,
    section_bbox: Tuple[float, float, float, float],
    section_page: int,
    phrases: List[Dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Find the phrase that matches a section heading among extracted phrases.

    Args:
        section_text: Section heading text.
        section_bbox: Bounding box of the section heading.
        section_page: Page number of the section heading (0-indexed, matching phrases).
        phrases: List of phrases with visual patterns.
        overlap_threshold: Minimum overlap ratio to qualify as a match.

    Returns:
        The matching phrase dictionary, or None if no match is found.
    """
    # ===== Step 1: Initialize best-match result =====
    best_match = None
    best_overlap = 0.0

    # ===== Step 2: Iterate over all phrases for matching =====
    for phrase in phrases:
        # Step 2.1: Check if the page number matches
        if phrase.get("page") != section_page:
            continue

        # Step 2.2: Prepare text for comparison (case-insensitive)
        phrase_text = phrase.get("phrase", "").strip()
        section_text_lower = section_text.lower()
        phrase_text_lower = phrase_text.lower()

        # ===== Step 3: Try multiple matching strategies =====
        # Strategy 1: Exact match
        if phrase_text_lower == section_text_lower:
            matched = True
        # Strategy 2: Section text starts with the phrase (for multi-word headings)
        elif section_text_lower.startswith(phrase_text_lower):
            matched = True
        # Strategy 3: Phrase starts with the section text (for concatenated phrases)
        elif phrase_text_lower.startswith(section_text_lower):
            matched = True
        # Strategy 4: Keyword partial match (e.g. headings like "1 INTRODUCTION")
        elif len(phrase_text.split()) >= 2 and any(
            word in section_text_lower for word in phrase_text_lower.split()
        ):
            matched = True
        # Strategy 5: Compare after removing spaces (for concatenated phrases)
        elif phrase_text_lower == section_text_lower.replace(" ", ""):
            matched = True
        else:
            matched = False

        if not matched:
            continue

        # ===== Step 4: Compute bounding box overlap =====
        phrase_bbox = phrase.get("bbox")
        if phrase_bbox:
            overlap = calculate_bbox_overlap(section_bbox, phrase_bbox)

            # Step 4.1: Update best match if overlap is higher and above threshold
            if overlap > best_overlap and overlap >= overlap_threshold:
                best_overlap = overlap
                best_match = phrase

    return best_match


def merge_phrase_group(
    phrase_group: List[Dict[str, Any]], merged_id: int
) -> Dict[str, Any]:
    """
    Merge a group of phrases with the same visual attributes into a single phrase.

    Args:
        phrase_group: List of phrases to merge.
        merged_id: ID to assign to the merged phrase.

    Returns:
        The merged phrase object.
    """
    if len(phrase_group) == 1:
        # Single phrase -- just update its ID
        result = phrase_group[0].copy()
        result["id"] = merged_id
        return result

    # Merge text
    merged_text = " ".join([phrase["phrase"] for phrase in phrase_group])

    # Merge bounding boxes (take the enclosing bounding box of all phrases)
    x0 = min(phrase["bbox"][0] for phrase in phrase_group)
    y0 = min(phrase["bbox"][1] for phrase in phrase_group)
    x1 = max(phrase["bbox"][2] for phrase in phrase_group)
    y1 = max(phrase["bbox"][3] for phrase in phrase_group)
    merged_bbox = (x0, y0, x1, y1)

    # Use attributes from the first phrase (all phrases should share them)
    first_phrase = phrase_group[0]

    # Determine if the merged phrase is centered
    # Without page width info, fall back to the first phrase's original value
    merged_is_center = first_phrase["is_center"]

    # Determine if the merged phrase starts with a digit
    merged_num_st = 1 if merged_text and merged_text[0].isdigit() else 0

    # Determine if the merged phrase is all caps
    merged_all_cap = 1 if merged_text.isupper() and merged_text.isalpha() else 0

    return {
        "id": merged_id,
        "phrase": merged_text,
        "bbox": merged_bbox,
        "page": first_phrase["page"],
        "font": first_phrase["font"],
        "size": first_phrase["size"],
        "bold": first_phrase["bold"],
        "is_underline": first_phrase["is_underline"],
        "all_cap": merged_all_cap,
        "num_st": merged_num_st,
        "is_center": merged_is_center,
    }
