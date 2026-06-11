def rule_page1_period_line_with_month_and_year(doc: dict) -> list[dict]:
    """Match page-1 spans containing a month name and year, useful for all filing types."""
    import re
    try:
        month = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(month + r'.*\b20\d{2}\b', (span.get("text") or "") + " " + (span.get("text_span") or ""))
        ]
    except Exception:
        return []
