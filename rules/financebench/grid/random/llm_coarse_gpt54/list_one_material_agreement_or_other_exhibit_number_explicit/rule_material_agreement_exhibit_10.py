def rule_material_agreement_exhibit_10(doc: dict) -> list[dict]:
    """Match spans containing Exhibit 10.x references, the most common material agreement exhibit family."""
    import re
    out = []
    pat = re.compile(r"\bExhibit\s+10(?:\.\d+)?[A-Za-z]?\b", re.I)
    try:
        for span in doc.get("texts", []):
            if pat.search(span.get("text") or ""):
                out.append(span)
    except Exception:
        return []
    return out
