def rule_page1_h2_address_headers(doc: dict) -> list[dict]:
    """Match H2 page-1 section headers whose text is itself the address or ZIP."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level != "H2":
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                out.append(span)
            elif re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low):
                out.append(span)
        return out
    except Exception:
        return []
