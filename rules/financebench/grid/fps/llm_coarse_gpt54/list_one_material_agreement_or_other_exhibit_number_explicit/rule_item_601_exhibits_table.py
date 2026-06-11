def rule_item_601_exhibits_table(doc: dict) -> list[dict]:
    """Match exhibit tables under Item 9.01 / Item 15 / Exhibit Index sections."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            label = span.get("label", "")
            if label == "table":
                if re.search(r'item\s*9\.01', path, re.I) or re.search(r'item\s*15', path, re.I) or re.search(r'exhibit index', path, re.I) or re.search(r'financial statements and exhibits', path, re.I):
                    out.append(span)
                elif re.search(r'\|\s*exhibit', text, re.I) and re.search(r'\|\s*description', text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
