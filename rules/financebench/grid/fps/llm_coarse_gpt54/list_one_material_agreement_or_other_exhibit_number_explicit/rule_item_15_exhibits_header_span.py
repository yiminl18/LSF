def rule_item_15_exhibits_header_span(doc: dict) -> list[dict]:
    """Match section headers for Item 15 Exhibits and Financial Statement Schedules."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                text = span.get("text", "") or ""
                path = (span.get("structure", {}) or {}).get("path_text", "") or ""
                if re.search(r'item\s*15', text, re.I) or re.search(r'item\s*15', path, re.I):
                    if re.search(r'exhibit', text + " " + path, re.I):
                        out.append(span)
    except Exception:
        return []
    return out
