def rule_text_span_reporting_period_page1(doc: dict) -> list[dict]:
    """Match page-1 text spans whose text itself contains the reporting period phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and span.get("page_no") == 1
            and any(k in (span.get("text") or "").lower() for k in [
                "fiscal year ended",
                "quarterly period ended",
                "quarter ended",
                "date of report",
                "date of earliest event reported"
            ])
        ]
    except Exception:
        return []
