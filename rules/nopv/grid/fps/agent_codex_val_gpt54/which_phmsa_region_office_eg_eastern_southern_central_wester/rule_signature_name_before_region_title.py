def rule_signature_name_before_region_title(doc: dict) -> list[dict]:
    """Match a signature name line that is immediately followed by the Director/region title line."""
    try:
        import re

        name_re = re.compile(r"^[A-Z][A-Za-z.\-'\s,]+$")
        title_re = re.compile(
            r"\b(?:Acting\s+)?Director\b[^\n]{0,120}\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )

        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            next_span = texts[i + 1]
            text = (span.get("text") or "").strip()
            next_text = (next_span.get("text") or "").strip()
            if span.get("page_no", 0) < 2:
                continue
            if next_span.get("page_no") != span.get("page_no"):
                continue
            if name_re.match(text) and title_re.search(next_text):
                out.extend([span, next_span])
        return out
    except Exception:
        return []
