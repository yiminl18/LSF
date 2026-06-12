def rule_contents_page_range_early(doc: dict) -> list[dict]:
    """Match early-document contents spans where the currency/coin table is usually referenced."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            p = span.get("page_no")
            txt = (span.get("text") or "")
            if p is not None and p <= 15 and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
