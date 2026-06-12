def rule_annual_rate_gdp_sentence(doc: dict) -> list[dict]:
    """Match prose spans stating real GDP growth at an annual rate for a quarter."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"real\s+GDP.*?\b\d+\.\d+\s*percent\b.*?annual rate", txt, re.I) or \
               re.search(r"\b\d+\.\d+\s*percent\b.*?annual rate.*?real\s+GDP", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
