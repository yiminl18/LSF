def rule_most_recent_reading_numeric(doc: dict) -> list[dict]:
    """Match spans with most recent/latest wording and a 2-3 digit decimal number."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"(most recent|latest|recent|final|preliminary)", (span.get("text") or ""), re.I)
            and re.search(r"\b\d{2,3}\.?\d*\b", (span.get("text") or ""))
        ]
    except Exception:
        return []
