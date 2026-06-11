def rule_item_15_exhibits_text_lines(doc: dict) -> list[dict]:
    """Match text lines under Item 15 that list exhibits outside a formal table."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r'item\s*15', path, re.I):
                if re.search(r'exhibit', text, re.I) or re.search(r'^\s*(99(\.\d+)?|104|10(\.\d+)?|4(\.\d+)?)\b', text):
                    out.append(span)
    except Exception:
        return []
    return out
