def rule_page1_bold_short_address_fragments(doc: dict) -> list[dict]:
    """Match short bold page-1 spans that are likely split address fragments."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1 or s.get("bold") != 1:
                continue
            txt = (s.get("text") or "").strip()
            low = txt.lower()
            if len(txt) > 80:
                continue
            if re.search(r"\b\d{1,6}\b", txt) and (
                re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low)
                or re.fullmatch(r"\d{5}(?:-\d{4})?", txt)
            ):
                out.append(s)
        return out
    except Exception:
        return []
