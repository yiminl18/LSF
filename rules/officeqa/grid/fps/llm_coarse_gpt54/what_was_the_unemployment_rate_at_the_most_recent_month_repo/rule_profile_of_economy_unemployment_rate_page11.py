def rule_profile_of_economy_unemployment_rate_page11(doc: dict) -> list[dict]:
    """Match page 11 spans in Profile of the Economy around the unemployment rate chart/text."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if page == 11 and "Profile of the Economy" in path:
                if re.search(r"Unemployment Rate|unemployment rate", text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
