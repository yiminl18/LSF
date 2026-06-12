def rule_profile_economy_federal_budget_and_debt_subsection_text(doc: dict) -> list[dict]:
    """Match text spans whose path_text includes the Federal Budget and Debt subsection."""
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if span.get("label") == "text" and "federal budget and debt" in path and "gdp" in txt:
                out.append(span)
    except Exception:
        return []
    return out
