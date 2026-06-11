def rule_page1_date_of_report_parenthetical(doc: dict) -> list[dict]:
    """Match page-1 8-K date lines with both filing date and earliest event date in parentheses."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'Date of Report.*' + month + r'\s+\d{1,2},\s+\d{4}.*\(' + month + r'\s+\d{1,2},\s+\d{4}\)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
