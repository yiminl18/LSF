def rule_exact_date_of_report(doc: dict) -> list[dict]:
    """Match spans containing 'Date of Report' or 'Date of earliest event reported' for 8-K event dates."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bDate of Report\b|\bDate of earliest event reported\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
