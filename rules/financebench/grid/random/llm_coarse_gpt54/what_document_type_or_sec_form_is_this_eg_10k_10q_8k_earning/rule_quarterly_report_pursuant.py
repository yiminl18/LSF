def rule_quarterly_report_pursuant(doc: dict) -> list[dict]:
    """Match spans containing QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d)."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in (span.get("text") or "").upper()
        ]
    except Exception:
        return []
