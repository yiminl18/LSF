def rule_profile_of_economy_recent_month_unemployment(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention unemployment rate together with a recent month name."""
    import re
    out = []
    months = r"January|February|March|April|May|June|July|August|September|October|November|December"
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" in path and re.search(r"\bunemployment rate\b", text, re.I) and re.search(months, text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
