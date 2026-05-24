def rule_page1_address_with_zip_no_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing full address and ZIP even when no explicit address label is present."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                if re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                    out.append(s)
        return out
    except Exception:
        return []
