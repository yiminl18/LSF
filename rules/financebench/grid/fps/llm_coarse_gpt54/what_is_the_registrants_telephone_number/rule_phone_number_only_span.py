def rule_phone_number_only_span(doc: dict) -> list[dict]:
    """Match spans whose text is mostly just a phone number."""
    try:
        import re
        out = []
        pat = re.compile(r"^\s*(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})\s*$")
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if pat.match(txt):
                out.append(span)
        return out
    except Exception:
        return []
