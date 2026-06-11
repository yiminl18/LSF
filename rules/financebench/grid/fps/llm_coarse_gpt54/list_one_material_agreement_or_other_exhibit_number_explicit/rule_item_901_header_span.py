def rule_item_901_header_span(doc: dict) -> list[dict]:
    """Match section headers for Item 9.01 Financial Statements and Exhibits."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                text = span.get("text", "") or ""
                path = (span.get("structure", {}) or {}).get("path_text", "") or ""
                if re.search(r'item\s*9\.01', text, re.I) or re.search(r'item\s*9\.01', path, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
