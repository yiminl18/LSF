def rule_exact_for_the_quarterly_period_ended(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'For the quarterly period ended'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bFor the quarterly period ended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
