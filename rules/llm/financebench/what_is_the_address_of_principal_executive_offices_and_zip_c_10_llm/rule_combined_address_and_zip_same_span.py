def rule_combined_address_and_zip_same_span(doc: dict) -> list[dict]:
    """Match spans where address and ZIP appear together in one span."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt) and re.search(r"\b\d{1,6}\b", txt):
                low = txt.lower()
                if re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                    out.append(span)
        return out
    except Exception:
        return []
