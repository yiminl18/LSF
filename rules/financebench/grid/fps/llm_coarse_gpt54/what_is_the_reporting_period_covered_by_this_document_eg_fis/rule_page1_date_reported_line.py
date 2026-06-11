def rule_page1_date_reported_line(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'reported' in the standard 8-K date line."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ("date of report" in (span.get("text") or "").lower() or "event reported" in (span.get("text") or "").lower())
        ]
    except Exception:
        return []
