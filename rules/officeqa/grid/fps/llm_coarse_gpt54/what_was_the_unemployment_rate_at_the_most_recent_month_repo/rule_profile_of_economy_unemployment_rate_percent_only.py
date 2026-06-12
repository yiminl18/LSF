def rule_profile_of_economy_unemployment_rate_percent_only(doc: dict) -> list[dict]:
    """Match unemployment-related spans with decimal percent values, regardless of exact phrasing."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"unemployment", text, re.I) and re.search(r"\b\d+\.\d\s*percent\b", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
