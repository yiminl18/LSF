def rule_profile_of_economy_unemployment_rate_numeric_band(doc: dict) -> list[dict]:
    """Match spans with unemployment rate values in the typical modern range 3.0-6.5 percent."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            m = re.search(r"(\d+\.\d)\s*percent", text, re.I)
            if m and re.search(r"unemployment rate", text, re.I):
                val = float(m.group(1))
                if 3.0 <= val <= 6.5:
                    out.append(span)
    except Exception:
        return []
    return out
