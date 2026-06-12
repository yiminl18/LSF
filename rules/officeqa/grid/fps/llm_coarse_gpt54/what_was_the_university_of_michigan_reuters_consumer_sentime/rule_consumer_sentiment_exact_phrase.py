def rule_consumer_sentiment_exact_phrase(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase consumer sentiment."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bconsumer sentiment\b", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
