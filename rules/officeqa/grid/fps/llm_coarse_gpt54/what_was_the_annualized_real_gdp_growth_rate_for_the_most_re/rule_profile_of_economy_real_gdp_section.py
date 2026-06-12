def rule_profile_of_economy_real_gdp_section(doc: dict) -> list[dict]:
    """Match spans in the Profile of the Economy section that discuss real GDP growth."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(r"\breal\s+(gross\s+domestic\s+product|GDP)\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
