def rule_profile_economy_fiscal_year_deficit_percent(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans mentioning fiscal year and deficit as percent of GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "fiscal year" in low
                and "deficit" in low
                and "percent of gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
