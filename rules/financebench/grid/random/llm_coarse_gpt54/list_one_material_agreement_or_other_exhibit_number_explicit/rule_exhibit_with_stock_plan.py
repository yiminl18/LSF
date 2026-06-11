def rule_exhibit_with_stock_plan(doc: dict) -> list[dict]:
    """Match spans containing an exhibit number near stock plan language."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I) and re.search(r"(stock incentive plan|equity incentive plan|employee stock purchase plan|stock purchase plan)", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
