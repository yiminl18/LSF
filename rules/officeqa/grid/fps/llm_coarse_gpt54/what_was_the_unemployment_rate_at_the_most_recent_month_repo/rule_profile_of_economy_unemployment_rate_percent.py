def rule_profile_of_economy_unemployment_rate_percent(doc: dict) -> list[dict]:
    """Match Profile of the Economy text spans containing 'unemployment rate' and a percent value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" in path and re.search(r"\bunemployment rate\b", text, re.I) and re.search(r"\b\d+\.\d\s*percent\b", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
