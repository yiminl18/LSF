def rule_profile_economy_federal_budget_deficit_paragraph(doc: dict) -> list[dict]:
    """Match text spans in Profile of the Economy mentioning the federal budget deficit and GDP."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") in {"text", "section_header"}
                and "profile of the economy" in path.lower()
                and "federal budget deficit" in txt.lower()
                and "percent of gdp" in txt.lower()
            ):
                out.append(span)
    except Exception:
        return []
    return out
