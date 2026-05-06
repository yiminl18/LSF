def rule_page1_address_like_excluding_long_narrative(doc: dict) -> list[dict]:
    """Match concise page-1 address-like spans while excluding long narrative cover text."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).strip()
            low = txt.lower()
            if len(txt) > 180:
                continue
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt) and re.search(r"\b\d{1,6}\b", txt):
                if re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                    out.append(s)
        return out
    except Exception:
        return []
