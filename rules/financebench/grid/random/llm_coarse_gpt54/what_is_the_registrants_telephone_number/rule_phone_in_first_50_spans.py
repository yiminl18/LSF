def rule_phone_in_first_50_spans(doc: dict) -> list[dict]:
    """Match likely phone-number spans within the first 50 spans of the filing."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in (doc.get("texts", []) or [])[:50]:
            if phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
