def rule_page1_address_like_and_not_phone(doc: dict) -> list[dict]:
    """Match page-1 address-like spans while excluding phone-number-only spans."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'\(\d{3}\)|\d{3}[-/]\d{3}', t):
                continue
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
