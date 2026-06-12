def rule_profile_of_economy_labor_markets_unemployment_rate(doc: dict) -> list[dict]:
    """Match labor-market spans that explicitly state the unemployment rate value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"Labor Markets|Employment and unemployment", path, re.I) and re.search(r"unemployment rate", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
