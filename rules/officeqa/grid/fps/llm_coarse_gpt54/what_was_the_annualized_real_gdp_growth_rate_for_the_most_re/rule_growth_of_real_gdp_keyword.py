def rule_growth_of_real_gdp_keyword(doc: dict) -> list[dict]:
    """Match spans explicitly mentioning 'Growth of Real GDP' or equivalent GDP growth phrasing."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"Growth of Real GDP|growth in real GDP|real GDP rose|real GDP growth", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
