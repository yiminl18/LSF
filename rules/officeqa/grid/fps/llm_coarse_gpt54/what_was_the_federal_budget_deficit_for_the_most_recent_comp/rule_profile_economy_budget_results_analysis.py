def rule_profile_economy_budget_results_analysis(doc: dict) -> list[dict]:
    """Match Profile of the Economy analysis text that summarizes budget deficit as a share of GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and ("budget deficit" in low or "federal deficit" in low)
                and ("share of gdp" in low or "percent of gdp" in low)
            ):
                out.append(span)
    except Exception:
        return []
    return out
