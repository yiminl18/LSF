def rule_page1_address_like_with_known_cover_labels(doc: dict) -> list[dict]:
    """Match spans where address-like text co-occurs with cover-page labels in text_span."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "").strip()
            tsp = (s.get("text_span") or "").lower()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                if "address of principal executive offices" in tsp or "zip code" in tsp or "telephone number" in tsp:
                    out.append(s)
        return out
    except Exception:
        return []
