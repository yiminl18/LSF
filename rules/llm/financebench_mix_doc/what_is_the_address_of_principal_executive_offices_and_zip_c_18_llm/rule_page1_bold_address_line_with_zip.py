def rule_page1_bold_address_line_with_zip(doc: dict) -> list[dict]:
    """Match bold page-1 spans whose text starts with a street number and contains a ZIP/postal code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'^\d{1,5}\s', text) and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', text) or
                re.search(r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
