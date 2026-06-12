def rule_profile_of_economy_unemployment_rate_stood_at(doc: dict) -> list[dict]:
    """Match spans using the phrasing 'unemployment rate stood at ... percent'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"unemployment rate stood at \d+\.\d\s*percent", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
