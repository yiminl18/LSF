def rule_text_spans_with_net_income_and_year(doc: dict) -> list[dict]:
    """Match narrative text spans that mention net income/earnings/loss and a fiscal year."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", s.get("text", "") or "", re.I)
            and re.search(r"\b20\d{2}\b|fiscal year", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
