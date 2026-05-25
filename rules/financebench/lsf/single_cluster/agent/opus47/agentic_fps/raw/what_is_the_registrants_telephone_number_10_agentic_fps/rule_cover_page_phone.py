import re

_PHONE_RE = re.compile(
    r"\(\d{3}\)\s*\d{3}\s*-\s*\d{4}"          # (206) 266-1000, (763)764-7600
    r"|\d{3}\s*-\s*\d{3}\s*-\s*\d{4}"         # 607-974-9000
    r"|\+\d{1,3}\s+\d{3}\s+\d{4,}"            # +44 117 9753200
    r"|\b\d{10}\b"                            # 6122911000 (bare 10-digit)
)


def rule_cover_page_phone(doc: dict) -> list[dict]:
    """Page-1 spans whose text contains a phone-number-like digit pattern."""
    return [
        span
        for span in doc.get("texts", [])
        if span.get("page_no") == 1
        and _PHONE_RE.search(span.get("text") or "")
    ]
