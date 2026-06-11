def rule_page1_city_state_zip_line(doc: dict) -> list[dict]:
    """Match page-1 spans that look like city/state/zip lines in the cover address block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and (
                re.search(r'^[A-Z][A-Z\s\.\'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$', text) or
                re.search(r'^[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$', text) or
                re.search(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d{4,}$', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
