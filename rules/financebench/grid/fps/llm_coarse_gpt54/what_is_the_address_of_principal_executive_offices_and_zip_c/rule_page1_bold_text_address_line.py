def rule_page1_bold_text_address_line(doc: dict) -> list[dict]:
    """Match bold page-1 text spans that are address lines even if not labeled as section headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
