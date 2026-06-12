def rule_profile_of_economy_introduction_unemployment(doc: dict) -> list[dict]:
    """Match introduction spans in Profile of the Economy that mention unemployment."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Introduction" in path and "Profile of the Economy" in path and re.search(r"unemployment", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
