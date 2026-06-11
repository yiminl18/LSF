def rule_page1_address_like_with_font_emphasis(doc: dict) -> list[dict]:
    """Match page-1 address-like spans with larger font sizes, common in cover-page registrant blocks."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if float(span.get("size") or 0) < 7:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
