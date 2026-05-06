def rule_page1_address_or_zip_combined_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing both address and zip code labeling language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices', text, re.I) and re.search(r'zip code', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
