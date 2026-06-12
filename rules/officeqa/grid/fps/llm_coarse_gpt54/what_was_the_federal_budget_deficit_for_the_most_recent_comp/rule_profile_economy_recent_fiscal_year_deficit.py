def rule_profile_economy_recent_fiscal_year_deficit(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans with 'fiscal year 20xx' and a deficit percent of GDP."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") != "text" or "profile of the economy" not in path:
                continue
            if re.search(r"fiscal year\s+(19|20)\d{2}", txt.lower()) and re.search(r"\d+(\.\d+)?\s+percent of gdp", txt.lower()):
                out.append(span)
    except Exception:
        return []
    return out
