def rule_profile_of_economy_unemployment_rate_latest_level(doc: dict) -> list[dict]:
    """Match spans describing unemployment as the latest or current level with a percent."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"unemployment rate", text, re.I) and re.search(r"\d+\.\d\s*percent", text, re.I):
                if re.search(r"current|latest|stood at|edged down|dipped to|declined to|reached", text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
