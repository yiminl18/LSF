import re

_PHONE_RE = re.compile(
    r"\(\d{3}\)\s*\d{3}[\-‐‑‒–—\s]\d{4}"
    r"|\b\d{3}-\d{3}-\d{4}\b"
    r"|\+\d{1,3}(?:[\s\d]){6,}\d"
)


def rule_registrant_telephone_cover_page(doc: dict) -> list[dict]:
    """Cover-page (page 1) spans that mention 'telephone' or contain a phone-like number."""
    out = []
    for span in doc.get("texts", []):
        if span.get("page_no") != 1:
            continue
        text = (span.get("text") or "")
        if "telephone" in text.lower() or _PHONE_RE.search(text):
            out.append(span)
    return out
