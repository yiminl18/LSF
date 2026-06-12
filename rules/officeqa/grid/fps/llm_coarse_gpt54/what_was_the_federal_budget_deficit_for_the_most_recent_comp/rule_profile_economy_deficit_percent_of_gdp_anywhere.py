def rule_profile_economy_deficit_percent_of_gdp_anywhere(doc: dict) -> list[dict]:
    """Match any span in Profile of the Economy that contains both deficit and percent of GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "profile of the economy" in path and "deficit" in txt and "percent of gdp" in txt:
                out.append(span)
    except Exception:
        return []
    return out
