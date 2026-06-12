def rule_profile_of_economy_gdp_growth_numeric_only(doc: dict) -> list[dict]:
    """Match spans with GDP growth numeric statements even if wording varies."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"(GDP|gross domestic product)", txt, re.I) and re.search(r"\b\d+\.\d+\b", txt) and re.search(r"(rose|grew|growth|accelerated|continued|advance|pace|annual rate)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
