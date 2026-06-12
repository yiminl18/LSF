def rule_profile_economy_deficit_share_of_economy(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans describing the deficit as a share of the economy/GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit" in txt
                and ("share of the economy" in txt or "share of gdp" in txt or "percent of gdp" in txt)
            ):
                out.append(span)
    except Exception:
        return []
    return out
