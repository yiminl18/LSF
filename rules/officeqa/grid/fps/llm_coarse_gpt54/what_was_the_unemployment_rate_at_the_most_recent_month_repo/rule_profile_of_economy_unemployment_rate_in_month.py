def rule_profile_of_economy_unemployment_rate_in_month(doc: dict) -> list[dict]:
    """Match spans saying 'In <Month> ... unemployment rate ... percent'."""
    import re
    out = []
    months = r"January|February|March|April|May|June|July|August|September|October|November|December"
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(rf"In {months}.*unemployment rate.*\d+\.\d\s*percent", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
