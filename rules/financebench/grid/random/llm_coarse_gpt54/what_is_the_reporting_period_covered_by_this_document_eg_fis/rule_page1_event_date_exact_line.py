def rule_page1_event_date_exact_line(doc: dict) -> list[dict]:
    """Match body spans on page 1 that are exactly the 8-K event-date line."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.fullmatch(r'Date of Report \(Date of earliest event reported\): .*', (span.get("text") or "").strip(), re.I)
        ]
    except Exception:
        return []
