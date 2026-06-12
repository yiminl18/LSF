def rule_profile_economy_gdp_percent_with_fiscal_year(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans with a fiscal year and a GDP percentage, likely the answer sentence."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "profile of the economy" not in path:
                continue
            if re.search(r"fiscal year\s+\d{4}", low) and re.search(r"\d+(\.\d+)?\s+percent of gdp", low):
                out.append(span)
    except Exception:
        return []
    return out
