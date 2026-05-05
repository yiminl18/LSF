def rule_for_the_quarterly_period_end_capitalized(doc: dict) -> list[dict]:
    """Match spans using capitalized 'For the Quarterly Period Ended' wording."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the quarterly period ended" in ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
        ]
    except Exception:
        return []
