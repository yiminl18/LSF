def rule_address_header_text_itself(doc: dict) -> list[dict]:
    """Match section headers whose own text is the address."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                out.append(span)
        return out
    except Exception:
        return []
