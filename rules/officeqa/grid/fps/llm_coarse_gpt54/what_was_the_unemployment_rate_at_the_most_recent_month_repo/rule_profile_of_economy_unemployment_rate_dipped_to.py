def rule_profile_of_economy_unemployment_rate_dipped_to(doc: dict) -> list[dict]:
    """Match spans using the phrasing 'unemployment rate dipped to ... percent'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"unemployment rate dipped to \d+\.\d\s*percent", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
