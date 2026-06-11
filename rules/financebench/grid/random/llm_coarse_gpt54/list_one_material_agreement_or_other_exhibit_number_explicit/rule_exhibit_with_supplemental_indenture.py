def rule_exhibit_with_supplemental_indenture(doc: dict) -> list[dict]:
    """Match spans containing an exhibit number near supplemental indenture language."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I) and re.search(r"\bsupplemental indenture\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
