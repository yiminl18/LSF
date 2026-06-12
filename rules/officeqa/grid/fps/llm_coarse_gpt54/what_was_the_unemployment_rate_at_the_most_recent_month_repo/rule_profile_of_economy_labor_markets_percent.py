def rule_profile_of_economy_labor_markets_percent(doc: dict) -> list[dict]:
    """Match labor-market discussion spans in Profile of the Economy that contain unemployment percentages."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" not in path:
                continue
            if re.search(r"Labor Markets|Employment and unemployment|Unemployment Rate", path, re.I) and re.search(r"\b\d+\.\d\s*percent\b", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
