def rule_page1_bold_phone_number(doc: dict) -> list[dict]:
    """Match bold page-1 spans that look like phone numbers."""
    try:
        import re
        out = []
        pat = re.compile(r"^.*(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12}).*$")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("bold") == 1:
                txt = span.get("text") or ""
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
