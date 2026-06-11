def rule_page1_same_span_address_and_zip(doc: dict) -> list[dict]:
    """Match page-1 spans that contain both the address and zip code in the same span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+.*\b\d{5}(?:-\d{4})?\b', t) or re.search(r'\bone\b.*\b\d{5}(?:-\d{4})?\b', t, re.I):
                if re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
