def rule_page1_zip_code_near_address(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning zip code together with address-related text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'zip code', t, re.I) and re.search(r'address|principal executive offices', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
