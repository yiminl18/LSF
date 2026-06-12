def rule_gdp_annual_rate_with_quarter_reference(doc: dict) -> list[dict]:
    """Match spans with quarter reference plus annual-rate GDP growth wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"(first|second|third|fourth)\s+quarter", txt, re.I) and \
               re.search(r"annual rate", txt, re.I) and \
               re.search(r"real\s+GDP|gross\s+domestic\s+product|GDP", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
