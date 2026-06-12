def rule_modern_debt_suspension_with_date(doc: dict) -> list[dict]:
    """Match spans that state suspension until a specific date, the modern answer pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"suspended until [A-Z][a-z]+ \d{1,2}, \d{4}", text):
                out.append(span)
        return out
    except Exception:
        return []
