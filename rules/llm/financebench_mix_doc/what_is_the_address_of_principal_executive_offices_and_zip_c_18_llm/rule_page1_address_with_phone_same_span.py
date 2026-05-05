def rule_page1_address_with_phone_same_span(doc: dict) -> list[dict]:
    """Match page-1 spans where address and phone number appear together."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'\d{1,5} .+', text) and re.search(r'\(?\+?\d[\d\-\)\( ]{6,}', text):
                if re.search(r'address|principal executive offices|zip code', text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
