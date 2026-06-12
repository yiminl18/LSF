def rule_profile_economy_budget_deficit_and_debt_paragraph(doc: dict) -> list[dict]:
    """Match Profile of the Economy paragraphs mentioning both budget deficit and debt."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit" in low
                and "debt" in low
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
