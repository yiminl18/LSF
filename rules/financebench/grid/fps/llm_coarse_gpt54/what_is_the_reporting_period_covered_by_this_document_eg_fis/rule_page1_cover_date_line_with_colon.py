def rule_page1_cover_date_line_with_colon(doc: dict) -> list[dict]:
    """Match page-1 cover date lines that use a colon before the date."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ":" in (span.get("text") or "")
            and any(k in (span.get("text") or "").lower() for k in [
                "date of report", "date of earliest event reported"
            ])
        ]
    except Exception:
        return []
