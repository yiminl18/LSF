def rule_phone_in_text_span(doc: dict) -> list[dict]:
    """Match text-label spans containing a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            if phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
