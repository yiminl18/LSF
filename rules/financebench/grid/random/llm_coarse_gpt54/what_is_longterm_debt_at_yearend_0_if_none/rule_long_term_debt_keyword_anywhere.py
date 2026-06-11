def rule_long_term_debt_keyword_anywhere(doc: dict) -> list[dict]:
    """Match any span containing the exact phrase 'long-term debt' or close variants."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\blong[\-\s]?term debt\b", (span.get("text", "") or ""), re.I)
        ]
    except Exception:
        return []
