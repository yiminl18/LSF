def rule_quarterly_report_phrase(doc: dict) -> list[dict]:
    """Match spans containing QUARTERLY REPORT PURSUANT..., indicating Form 10-Q."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in ((span.get("text") or "").upper())
        ]
    except Exception:
        return []
