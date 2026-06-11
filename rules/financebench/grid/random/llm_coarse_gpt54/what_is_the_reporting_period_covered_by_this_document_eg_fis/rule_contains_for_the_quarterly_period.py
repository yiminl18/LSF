def rule_contains_for_the_quarterly_period(doc: dict) -> list[dict]:
    """Match any span containing 'For the quarterly period'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bFor the quarterly period\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
