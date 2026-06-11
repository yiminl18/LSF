def rule_annual_report_page1(doc: dict) -> list[dict]:
    """Match page-1 spans containing ANNUAL REPORT, typically used on Form 10-K covers."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and "ANNUAL REPORT" in (span.get("text") or "").upper()
        ]
    except Exception:
        return []
