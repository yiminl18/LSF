def rule_exhibit_table_header_text(doc: dict) -> list[dict]:
    """Match tables whose markdown text contains Exhibit/Description headers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'exhibit\s*(no\.|number)?', text, re.I) and re.search(r'description', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
