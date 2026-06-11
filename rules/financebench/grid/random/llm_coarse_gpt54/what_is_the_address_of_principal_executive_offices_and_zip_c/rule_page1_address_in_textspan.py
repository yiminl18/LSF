def rule_page1_address_in_textspan(doc: dict) -> list[dict]:
    """Match page-1 spans where text_span contains address/zip labels, often with address in text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            ts = span.get("text_span") or ""
            if re.search(r'address of principal executive offices|address and telephone number.*principal executive offices|address of principal executive offices and zip code', ts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
