def rule_profile_of_economy_unemployment_paragraph(doc: dict) -> list[dict]:
    """Match text spans in the Profile of the Economy section mentioning the unemployment rate."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if "Profile of the Economy" in path and re.search(r"\bunemployment rate\b", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
