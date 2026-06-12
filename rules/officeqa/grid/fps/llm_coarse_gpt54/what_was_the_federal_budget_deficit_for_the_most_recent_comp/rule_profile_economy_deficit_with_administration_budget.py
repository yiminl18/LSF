def rule_profile_economy_deficit_with_administration_budget(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention the Administration budget and deficit/GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "administration" in low
                and "budget" in low
                and "deficit" in low
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
