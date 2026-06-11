def rule_page1_phone_number_pattern(doc: dict) -> list[dict]:
    """Match page-1 spans containing phone-number-like patterns."""
    try:
        import re
        out = []
        pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]\d{3}[-\s]\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = span.get("text") or ""
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
