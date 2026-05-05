def rule_page1_address_and_telephone_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing address-and-telephone office label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"address and telephone number.*principal executive offices", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
