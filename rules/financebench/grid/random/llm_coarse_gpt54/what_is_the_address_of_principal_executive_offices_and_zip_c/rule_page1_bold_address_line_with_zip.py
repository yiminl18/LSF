def rule_page1_bold_address_line_with_zip(doc: dict) -> list[dict]:
    """Match bold page-1 text spans that look like a street address ending in a ZIP/postal code."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\d{3,}.*\b(\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2})\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
