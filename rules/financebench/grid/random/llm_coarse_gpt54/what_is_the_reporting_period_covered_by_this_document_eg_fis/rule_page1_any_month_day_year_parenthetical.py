def rule_page1_any_month_day_year_parenthetical(doc: dict) -> list[dict]:
    """Match page-1 spans with a parenthetical date, common in 8-K event-date lines."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\(' + month + r'\s+\d{1,2},\s+\d{4}\)', (span.get("text") or "") + " " + (span.get("text_span") or ""))
        ]
    except Exception:
        return []
