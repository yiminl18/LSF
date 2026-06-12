def rule_profile_economy_deficit_with_fy_budget_projection(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention a fiscal-year budget projection and deficit percent."""
    out = []
    try:
        for span in doc.get("texts", []):
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and ("budget projects" in low or "budget projects the deficit" in low or "budget projects" in low)
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
