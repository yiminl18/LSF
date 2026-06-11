def rule_exhibit_10_in_table(doc: dict) -> list[dict]:
    """Match table spans containing Exhibit 10.x references."""
    import re
    out = []
    pat = re.compile(r"\bExhibit\s+10(?:\.\d+)?[A-Za-z]?\b", re.I)
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table" and pat.search(span.get("text") or ""):
                out.append(span)
    except Exception:
        return []
    return out
