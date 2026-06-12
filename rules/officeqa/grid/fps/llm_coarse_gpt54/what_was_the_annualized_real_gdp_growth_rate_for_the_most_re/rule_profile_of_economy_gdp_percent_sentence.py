def rule_profile_of_economy_gdp_percent_sentence(doc: dict) -> list[dict]:
    """Match Profile of the Economy prose containing a percent and GDP in the same span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(r"\b(GDP|gross domestic product)\b", txt, re.I) and re.search(r"\d+\.\d+\s*percent", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
