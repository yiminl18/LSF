import re


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_RE = "|".join(_MONTHS)

_MONTH_DAY_RE = rf"(?:{_MONTH_RE})\s+\d{{1,2}}(?:,\s*\d{{4}})?"
_DAY_YEAR_RE = r"(?:\d{1,2}(?:,\s*\d{4})?)"
_MONTH_YEAR_RE = rf"(?:{_MONTH_RE})\s+\d{{4}}"
_DATE_RE = rf"(?:{_MONTH_DAY_RE}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{1,2}}-\d{{1,2}}-\d{{2,4}})"

_RANGE_PATTERNS = [
    # The most common opener: "From <date> to/through <date>, ... inspected ..."
    re.compile(
        rf"\bFrom\s+({_DATE_RE})(?:,\s*)?\s+(?:to|through|[-–])\s+({_DATE_RE}|{_DAY_YEAR_RE})",
        re.IGNORECASE,
    ),
    # Openers that describe the inspection as occurring on "various dates between".
    re.compile(
        rf"\bOn\s+various dates between\s+({_MONTH_YEAR_RE}|{_DATE_RE})\s+and\s+({_MONTH_YEAR_RE}|{_DATE_RE})",
        re.IGNORECASE,
    ),
    # Openers like "Between July and October 2021, a representative ... inspected ...".
    re.compile(
        rf"\bBetween\s+({_MONTH_RE})\s+and\s+({_MONTH_RE})\s+\d{{4}}",
        re.IGNORECASE,
    ),
    # Openers like "From/For the weeks of July 29 to August 2, 2024, and ...".
    re.compile(
        r"\b(?:From|For)\s+the\s+weeks?\s+of\s+(.+?)(?=,\s*(?:a|an|the|one|representative|representatives|PHMSA)\b|\s+(?:a|an|the|one|representative|representatives|PHMSA)\b)",
        re.IGNORECASE,
    ),
    # Openers like "During the September 20 through 24, 2021 field inspection".
    re.compile(
        rf"\bDuring\s+the\s+((?:{_MONTH_RE})\s+\d{{1,2}}\s+(?:to|through|[-–])\s+(?:(?:{_MONTH_RE})\s+)?\d{{1,2}}(?:,\s*\d{{4}})?)\s+(?:field\s+)?inspection\b",
        re.IGNORECASE,
    ),
    # Openers like "During PHMSA's field inspection on April 10, 2025".
    re.compile(
        rf"\bDuring\s+(?:PHMSA(?:'s)?\s+)?(?:on-site\s+)?(?:field\s+)?inspection\s+on\s+({_DATE_RE})",
        re.IGNORECASE,
    ),
    # Single-date openers: "On May 10, 2023, ... inspected ..."
    re.compile(
        rf"\bOn\s+({_DATE_RE})",
        re.IGNORECASE,
    ),
]

_FALLBACK_OPENING_RE = re.compile(
    r"^\s*(?:From|On|During)\b.{0,260}?\binspect(?:ed|ion)\b",
    re.IGNORECASE | re.DOTALL,
)


def rule_phmsa_inspection_date_range(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def make_span(text: str, src: dict | None = None) -> dict:
            span = {"text": norm(text)}
            if src:
                for key in ("page_no", "line_no", "paragraph_no"):
                    if key in src:
                        span[key] = src[key]
            return span

        def maybe_add_span(text: str, src: dict | None = None) -> list[dict] | None:
            text = norm(text)
            if not text:
                return None

            for pat in _RANGE_PATTERNS:
                match = pat.search(text)
                if not match:
                    continue
                if pat is _RANGE_PATTERNS[2]:
                    return [make_span(match.group(0), src)]
                if pat is _RANGE_PATTERNS[3]:
                    return [make_span(match.group(1), src)]
                if pat is _RANGE_PATTERNS[4]:
                    return [make_span(match.group(1), src)]
                if pat is _RANGE_PATTERNS[5]:
                    return [make_span(match.group(1), src)]
                # Preserve the full matched range phrase for the other opener types.
                return [make_span(match.group(0), src)]

            fallback = _FALLBACK_OPENING_RE.search(text)
            if fallback and re.search(r"\b(?:PHMSA|representative|representatives|inspector)\b", fallback.group(0), re.IGNORECASE):
                return [make_span(fallback.group(0), src)]
            return None

        # Search the most local/topical containers first.
        for source_key, limit in (("pages", 2), ("paragraphs", 12), ("lines", 40)):
            items = [s for s in (doc.get(source_key) or []) if isinstance(s, dict)]
            for item in items[:limit]:
                found = maybe_add_span(item.get("text") or "", item)
                if found:
                    return found

        # Fallback to the top of the raw text if the document was split unusually.
        text = doc.get("text") or ""
        head = text[:4000] if isinstance(text, str) else ""
        found = maybe_add_span(head, None)
        if found:
            return found

        return []
    except Exception:
        return []
