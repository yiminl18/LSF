"""Stage A subroutine — regex-based rule specificity score (no LLM).

A high score (close to 1) means the rule encodes layout-specific patterns
(hardcoded page numbers, month names, page ranges) and is at risk of
overfitting the sampled docs.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Patterns ──────────────────────────────────────────────────────────────────

# `page_no == 39`, `page == 42`, `page_around_39`, `page 60`
_PAGE_EQ = re.compile(
    r"page[_ ]?(?:no|number)?\s*[=<>!]=?\s*\d+|"
    r"page_around_\d+|"
    r"\bpage\s+\d+\b",
    re.IGNORECASE,
)

# Month names (full or three-letter)
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)

# `pages_30_to_70`, `pages 30 to 70`
_PAGE_RANGE = re.compile(
    r"pages?_\d+_to_\d+|pages?\s+\d+\s+to\s+\d+",
    re.IGNORECASE,
)


def specificity_score(rule_source: str) -> float:
    """Return a score in [0, 0.8] from the rule's Python source.

    Higher means more specific (page/date/range constraints). The cap at 0.8
    keeps even very narrow rules from being fully zeroed in the utility score —
    they can still win if their proxy_acc and coverage are exceptional.
    """
    if not rule_source:
        return 0.0
    score = 0.0
    if _PAGE_EQ.search(rule_source):
        score += 0.5
    if _MONTH.search(rule_source):
        score += 0.3
    if _PAGE_RANGE.search(rule_source):
        score += 0.4
    return min(score, 0.8)


def specificity_from_file(rule_file: Path) -> float:
    """Read a rule's .py file and return its specificity score."""
    try:
        return specificity_score(rule_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return 0.0
