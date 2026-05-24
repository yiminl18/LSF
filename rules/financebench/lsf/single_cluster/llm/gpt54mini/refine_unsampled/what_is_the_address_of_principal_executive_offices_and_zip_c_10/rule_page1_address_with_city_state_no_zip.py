def rule_page1_address_with_city_state_no_zip(doc: dict) -> list[dict]:
    """Match page-1 spans containing street plus city/state but no ZIP, for split ZIP layouts."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                if re.search(r"\b[A-Z][a-zA-Z .'-]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", txt) and not re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                    out.append(s)
        return out
    except Exception:
        return []
