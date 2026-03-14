# -*- coding: utf-8 -*-
"""
Text utility functions.

Pure utility functions extracted from processing/feature_extract.py
to break the doc <-> structure circular dependency.
"""

import re


def starts_with_number(text: str) -> bool:
    """Check whether the text starts with a digit (ignoring leading whitespace)."""
    return bool(re.match(r"^\s*\d", text or ""))


def starts_with_letter(text: str) -> bool:
    """Check whether the text starts with a letter (ignoring leading whitespace)."""
    return bool(re.match(r"^\s*[A-Za-z]", text or ""))
