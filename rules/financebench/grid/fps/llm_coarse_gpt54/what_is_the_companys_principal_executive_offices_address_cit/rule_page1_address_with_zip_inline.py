def rule_page1_address_with_zip_inline(doc: dict) -> list[dict]:
    """Match page-1 spans where street, city/state, and zip all appear inline."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?', text):
                out.append(span)
        return out
    except Exception:
        return []
