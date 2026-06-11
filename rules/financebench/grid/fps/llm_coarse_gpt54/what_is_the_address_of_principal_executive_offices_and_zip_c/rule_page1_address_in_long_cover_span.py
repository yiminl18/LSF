def rule_page1_address_in_long_cover_span(doc: dict) -> list[dict]:
    """Match long page-1 cover spans that embed the full address in a single text block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            full = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'^\s*(?:[A-Z][A-Za-z&\., ]+)?\s*\d{1,6}\s+\S+', full) or re.search(r'\bone [A-Za-z]', full, re.I):
                if re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', full, re.I):
                    if re.search(r'address of principal executive offices|zip code|registrant.?s telephone number|i\.r\.s\. employer identification', full, re.I):
                        out.append(span)
        return out
    except Exception:
        return []
