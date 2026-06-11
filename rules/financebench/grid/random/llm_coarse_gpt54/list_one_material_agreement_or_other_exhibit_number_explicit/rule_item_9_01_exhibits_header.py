def rule_item_9_01_exhibits_header(doc: dict) -> list[dict]:
    """Match 8-K Item 9.01 Financial Statements and Exhibits headers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"item\s*9\.?01", txt, re.I) and re.search(r"financial statements and exhibits", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
