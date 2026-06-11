def rule_exact_quarterly_period_ended_span(doc: dict) -> list[dict]:
    """Match spans whose text directly states 'For the quarterly period ended ...'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the quarterly period ended" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
