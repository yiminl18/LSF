def rule_profile_of_economy_unemployment_rate_near_profile_heading(doc: dict) -> list[dict]:
    """Match unemployment spans occurring after the Profile of the Economy heading and before the next major section."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        start = None
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r"Profile of the Economy", span.get("text", "") or "", re.I):
                start = i
                break
        if start is None:
            return []
        for j in range(start + 1, len(texts)):
            span = texts[j]
            if j > start + 1 and span.get("label") == "section_header" and re.search(r"Federal Fiscal Operations|Financial Operations|International Statistics|Special Reports", span.get("text", "") or "", re.I):
                break
            text = span.get("text", "") or ""
            if span.get("label") == "text" and re.search(r"unemployment rate", text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
