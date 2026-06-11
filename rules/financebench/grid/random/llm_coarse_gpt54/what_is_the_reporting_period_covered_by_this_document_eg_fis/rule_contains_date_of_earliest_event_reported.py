def rule_contains_date_of_earliest_event_reported(doc: dict) -> list[dict]:
    """Match any span containing 'Date of earliest event reported'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bDate of earliest event reported\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
