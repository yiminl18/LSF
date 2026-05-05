def rule_exact_for_the_quarterly_period_ended(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'For the quarterly period ended'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the quarterly period ended" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
