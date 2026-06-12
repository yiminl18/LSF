def rule_profile_of_economy_unemployment_rate_with_month_and_lowest(doc: dict) -> list[dict]:
    """Match unemployment spans that mention a month and say it is the lowest since some earlier date."""
    import re
    out = []
    months = r"January|February|March|April|May|June|July|August|September|October|November|December"
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(rf"{months}.*unemployment rate.*lowest|unemployment rate.*{months}.*lowest", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
