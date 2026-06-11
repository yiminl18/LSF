def rule_exhibit_reference_inline(doc: dict) -> list[dict]:
    """Match spans that explicitly say an item is included as Exhibit X.Y hereto."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r'Exhibit\s+\d+(\.\d+)?', text, re.I) and re.search(r'hereto|incorporated herein by reference|included as Exhibit', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
