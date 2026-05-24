def rule_page1_text_contains_address_and_text_span_has_zip_label(doc: dict) -> list[dict]:
    """Match spans where the main text is address and nearby text_span carries zip code labeling."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "")
            if re.search(r'^\d{1,5}\s', text) and re.search(r'zip code', text_span, re.I):
                out.append(span)
        return out
    except Exception:
        return []
