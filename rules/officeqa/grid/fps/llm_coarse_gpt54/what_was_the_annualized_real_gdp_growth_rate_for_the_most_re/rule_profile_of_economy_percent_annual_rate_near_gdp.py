def rule_profile_of_economy_percent_annual_rate_near_gdp(doc: dict) -> list[dict]:
    """Match any span with percent + annual rate + GDP in Profile of the Economy."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(r"\d+\.\d+\s*percent.*annual rate", txt, re.I) and re.search(r"GDP|gross domestic product", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
