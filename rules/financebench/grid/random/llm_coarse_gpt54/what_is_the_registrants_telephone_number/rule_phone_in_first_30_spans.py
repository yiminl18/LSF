def rule_phone_in_first_30_spans(doc: dict) -> list[dict]:
    """Match likely phone-number spans appearing very early in the document."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in (doc.get("texts", []) or [])[:30]:
            if phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
