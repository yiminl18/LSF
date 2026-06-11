def rule_phone_number_in_textspan_only(doc: dict) -> list[dict]:
    """Match spans where the phone number appears in text_span rather than text."""
    try:
        import re
        out = []
        pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            tsp = span.get("text_span") or ""
            if not pat.search(txt) and pat.search(tsp):
                out.append(span)
        return out
    except Exception:
        return []
