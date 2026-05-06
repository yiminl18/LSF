def rule_exact_date_of_report(doc: dict) -> list[dict]:
    """Match spans containing the 8-K event-date lead phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "date of report (date of earliest event reported)" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
