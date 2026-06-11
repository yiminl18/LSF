def rule_contains_date_of_report(doc: dict) -> list[dict]:
    """Match any span containing 'Date of Report'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bDate of Report\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
