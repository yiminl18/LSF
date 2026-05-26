import re


_PHMSA_CONTEXT_RE = re.compile(
    r"\b(?:PHMSA|Pipeline and Hazardous Materials Safety Administration)\b",
    re.IGNORECASE,
)

_OPENING_PATTERNS = [
    # Most common form: "From <date clause> ... a representative(s) ..."
    re.compile(
        r"\b(From\s+.+?)(?:\s+of\s+the\s+(?:on-site\s+)?inspection)?\s*,\s*(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(From\s+.+?)(?:\s+of\s+the\s+(?:on-site\s+)?inspection)?\s+(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "On various dates between <date> and <date>, a representative(s) ..."
    re.compile(
        r"\b(On\s+various dates between\s+.+?)\s*,\s*(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(On\s+various dates between\s+.+?)\s+(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Less common but still an opening clause.
    re.compile(
        r"\b(Between\s+.+?)\s*,\s*(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(Between\s+.+?)\s+(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(During\s+.+?)\s*,\s*(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(During\s+.+?)\s+(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(On\s+.+?)\s*,\s*(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(On\s+.+?)\s+(?:a|an|the|one)?\s*(?:representative|representatives)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Fallback when the opener names PHMSA directly instead of a representative.
    re.compile(
        r"\b(From\s+.+?)\s*,\s*(?:PHMSA(?:['’]s)?|the Pipeline and Hazardous Materials Safety Administration)\b.*?\b(?:inspected|investigated|reviewed|conducted)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(From\s+.+?)\s+(?:PHMSA(?:['’]s)?|the Pipeline and Hazardous Materials Safety Administration)\b.*?\b(?:inspected|investigated|reviewed|conducted)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(On\s+.+?)\s*,\s*(?:PHMSA(?:['’]s)?|the Pipeline and Hazardous Materials Safety Administration)\b.*?\b(?:inspected|investigated|reviewed|conducted)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(On\s+.+?)\s+(?:PHMSA(?:['’]s)?|the Pipeline and Hazardous Materials Safety Administration)\b.*?\b(?:inspected|investigated|reviewed|conducted)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]


def rule_phmsa_inspection_date_range(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def make_span(text: str, src: dict | None = None) -> dict:
            span = {"text": norm(text)}
            if src:
                for key in ("page_no", "paragraph_no", "line_no"):
                    if key in src:
                        span[key] = src[key]
            return span

        def scan(text: str, src: dict | None = None) -> list[dict] | None:
            text = norm(text)
            if not text or not _PHMSA_CONTEXT_RE.search(text):
                return None

            for pat in _OPENING_PATTERNS:
                match = pat.search(text)
                if match:
                    return [make_span(match.group(1), src)]
            return None

        # The answer is almost always in the early opening paragraph(s), so search
        # the structured containers from most local to least local.
        for source_key, limit in (("paragraphs", 16), ("lines", 48), ("pages", 2)):
            items = [s for s in (doc.get(source_key) or []) if isinstance(s, dict)]
            for item in items[:limit]:
                found = scan(item.get("text") or "", item)
                if found:
                    return found

        text = doc.get("text") or ""
        if isinstance(text, str):
            found = scan(text[:6000], None)
            if found:
                return found

        return []
    except Exception:
        return []
