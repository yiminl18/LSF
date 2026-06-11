def rule_page1_cover_date_line_without_colon(doc: dict) -> list[dict]:
    """Match page-1 cover date lines that omit the colon but still use the standard date-of-report wording."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and "date of report" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
