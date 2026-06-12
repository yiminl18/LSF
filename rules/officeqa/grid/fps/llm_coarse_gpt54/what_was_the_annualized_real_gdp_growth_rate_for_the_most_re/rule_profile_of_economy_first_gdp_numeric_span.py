def rule_profile_of_economy_first_gdp_numeric_span(doc: dict) -> list[dict]:
    """Return the first span in the economy profile that contains GDP and a decimal percent."""
    import re
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if "Profile of the Economy" in path and re.search(r"(GDP|gross domestic product)", txt, re.I) and re.search(r"\d+\.\d+\s*percent", txt, re.I):
                return [span]
        return []
    except Exception:
        return []
