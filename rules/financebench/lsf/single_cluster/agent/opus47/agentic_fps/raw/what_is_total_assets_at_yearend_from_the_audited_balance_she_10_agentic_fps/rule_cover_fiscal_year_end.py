def rule_cover_fiscal_year_end(doc: dict) -> list[dict]:
    '''Cover-page (page <= 3) spans declaring "For the fiscal year ended <date>" — anchors which year is year-end.'''
    out = []
    for span in doc.get("texts", []):
        page = span.get("page_no", 0)
        if page is None or page > 3:
            continue
        text = (span.get("text") or "")
        if not text:
            continue
        lower = text.lower()
        if "fiscal year ended" in lower and len(text) <= 240:
            out.append(span)
    return out
