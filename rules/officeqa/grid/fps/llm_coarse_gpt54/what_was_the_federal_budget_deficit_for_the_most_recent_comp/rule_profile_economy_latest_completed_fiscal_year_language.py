def rule_profile_economy_latest_completed_fiscal_year_language(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention the latest completed fiscal year explicitly."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "fiscal year 20" in low
                and "deficit" in low
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
