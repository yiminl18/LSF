# -*- coding: utf-8 -*-
"""
Font utilities: strip subset prefixes, detect bold, etc.
Shared by tree_reconstructor, template_clustering, and analysis scripts.
"""

import re
from typing import Optional


def clean_font_name(font_name: Optional[str]) -> str:
    """
    Strip the subset prefix from a PDF font name (e.g. ODMZRF+, BAAAAA+).

    When embedding a font subset, PDFs prepend six uppercase letters plus a '+'.
    This function removes that prefix so font names can be compared directly.
    """
    if not font_name:
        return ""
    return re.sub(r"^[A-Z]{6}\+", "", font_name.strip())


def is_bold_font(font_name: Optional[str]) -> bool:
    """Determine whether a font is bold based on its name."""
    if not font_name:
        return False
    font_lower = font_name.lower()
    # Explicit bold keywords
    bold_indicators = ["bold", "heavy", "black", "demibold", "semibold", "extrabold"]
    if any(indicator in font_lower for indicator in bold_indicators):
        return True
    # "-b" or "b-" pattern (e.g. "Arial-B")
    if "-b" in font_lower or "b-" in font_lower:
        return True
    if (
        font_lower.startswith("b")
        and len(font_lower) > 1
        and font_lower[1] in ["-", "_"]
    ):
        return True
    # TeX Computer Modern bold families: CMBX (Bold Extended), CMMIB (Math Italic Bold), CMSSBX (SS Bold Extended)
    clean = re.sub(r"^[a-z]{6}\+", "", font_lower)
    if re.match(r"cm(bx|mib|ssbx)", clean):
        return True
    return False
