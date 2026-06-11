def rule_page1_address_block_with_zip_in_text_span(doc: dict) -> list[dict]:
    """Match address spans whose neighboring text_span carries zip-code or address labels."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            ts = (span.get("text_span") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                if re.search(r'zip code|address of principal executive offices|address of principal executive offices and zip code', ts, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
