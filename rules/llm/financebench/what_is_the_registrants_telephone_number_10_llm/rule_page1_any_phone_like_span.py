def rule_page1_any_phone_like_span(doc: dict) -> list[dict]:
    """Match any page-1 span with a phone-like number, favoring high recall."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
