def rule_exhibit_number_line(doc: dict) -> list[dict]:
    """Match spans containing an explicit exhibit number pattern like 'Exhibit 10.1'."""
    import re
    out = []
    pat = re.compile(r"\bExhibit\s+(?:No\.\s*)?\d+(?:\.\d+)?[A-Za-z]?\b", re.I)
    try:
        for span in doc.get("texts", []):
            if pat.search(span.get("text") or ""):
                out.append(span)
    except Exception:
        return []
    return out
