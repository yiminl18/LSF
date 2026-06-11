def rule_exhibit_with_settlement_agreement(doc: dict) -> list[dict]:
    """Match spans containing an exhibit number near 'Settlement Agreement'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I) and re.search(r"\bsettlement agreement\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
