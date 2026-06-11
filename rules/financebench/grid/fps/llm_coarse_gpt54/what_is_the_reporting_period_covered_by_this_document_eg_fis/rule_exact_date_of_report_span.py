def rule_exact_date_of_report_span(doc: dict) -> list[dict]:
    """Match spans whose text directly states 'Date of Report' or 'Date of earliest event reported'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "date of report" in (span.get("text") or "").lower()
            or "date of earliest event reported" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
