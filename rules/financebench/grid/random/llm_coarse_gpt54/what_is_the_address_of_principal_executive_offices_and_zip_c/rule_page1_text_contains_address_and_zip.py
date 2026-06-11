def rule_page1_text_contains_address_and_zip(doc: dict) -> list[dict]:
    """Match page-1 spans that contain both a street number and a ZIP/postal code."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            if re.search(r'\d{2,}.*\b(\d{5}(?:-\d{4})?|BS30 ?8XP|[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2})\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
