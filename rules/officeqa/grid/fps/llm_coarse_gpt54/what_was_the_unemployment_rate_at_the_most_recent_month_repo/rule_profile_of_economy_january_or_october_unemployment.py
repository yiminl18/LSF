def rule_profile_of_economy_january_or_october_unemployment(doc: dict) -> list[dict]:
    """Match unemployment spans mentioning January or October, common report months in this collection."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Profile of the Economy" in path and re.search(r"\bunemployment rate\b", text, re.I):
                if re.search(r"\b(January|October|July|April)\b", text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
