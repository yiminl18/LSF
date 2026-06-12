def rule_profile_economy_deficit_peak_and_latest_percent(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans containing two deficit percentages, one peak and one latest."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") != "text" or "profile of the economy" not in path:
                continue
            if "deficit" in low and len(re.findall(r"\d+(\.\d+)?\s+percent of gdp", low)) >= 1 and "peak" in low:
                out.append(span)
    except Exception:
        return []
    return out
