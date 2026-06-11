def rule_phone_number_start_of_span(doc: dict) -> list[dict]:
    """Match spans whose text starts with a phone number."""
    try:
        import re
        out = []
        pat = re.compile(r"^\s*(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
