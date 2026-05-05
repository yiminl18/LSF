def rule_page1_address_block_with_zip_label_and_phone_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing both ZIP code and telephone labeling language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'zip code', text, re.I) and re.search(r'telephone number|area code', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
