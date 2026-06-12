def rule_profile_economy_percent_of_gdp_near_deficit(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans where 'percent of GDP' appears near 'deficit'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") != "text" or "profile of the economy" not in path:
                continue
            if re.search(r"deficit.{0,120}percent of gdp", low) or re.search(r"percent of gdp.{0,120}deficit", low):
                out.append(span)
    except Exception:
        return []
    return out
