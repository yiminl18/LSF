def rule_consumer_confidence_or_sentiment(doc: dict) -> list[dict]:
    """Match spans mentioning either consumer confidence or consumer sentiment."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bconsumer (confidence|sentiment)\b", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
