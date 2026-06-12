def rule_profile_of_economy_lowest_unemployment_sentence(doc: dict) -> list[dict]:
    """Match Profile of the Economy text spans saying unemployment is at the lowest level since some year."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" in path and re.search(r"lowest (figure|level).+unemployment rate|unemployment rate.+lowest", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
