def rule_real_gdp_growth_rate_numeric_phrase(doc: dict) -> list[dict]:
    """Match spans with a numeric percent near 'real GDP growth' phrasing."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"real\s+GDP\s+(growth|rose|advanced|accelerated|continued)", txt, re.I) and re.search(r"\b\d+\.\d+\s*percent\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
