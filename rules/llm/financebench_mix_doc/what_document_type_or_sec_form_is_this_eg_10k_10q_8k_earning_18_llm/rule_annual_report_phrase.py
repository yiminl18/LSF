def rule_annual_report_phrase(doc: dict) -> list[dict]:
    """Match spans containing ANNUAL REPORT PURSUANT..., indicating Form 10-K."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in ((span.get("text") or "").upper())
        ]
    except Exception:
        return []
