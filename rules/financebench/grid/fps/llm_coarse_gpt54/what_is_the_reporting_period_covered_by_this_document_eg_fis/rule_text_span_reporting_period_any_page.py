def rule_text_span_reporting_period_any_page(doc: dict) -> list[dict]:
    """Match any text span containing reporting-period language, regardless of page."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
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
