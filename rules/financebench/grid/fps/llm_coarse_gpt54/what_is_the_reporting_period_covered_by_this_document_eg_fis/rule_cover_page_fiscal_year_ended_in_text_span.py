def rule_cover_page_fiscal_year_ended_in_text_span(doc: dict) -> list[dict]:
    """Match spans whose text_span metadata contains the reporting period phrase on the cover page."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and any(k in (span.get("text_span") or "").lower() for k in [
                "fiscal year ended",
                "quarterly period ended",
                "date of report",
                "date of earliest event reported"
            ])
        ]
    except Exception:
        return []
