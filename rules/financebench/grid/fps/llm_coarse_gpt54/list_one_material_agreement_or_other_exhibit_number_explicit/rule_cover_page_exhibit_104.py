def rule_cover_page_exhibit_104(doc: dict) -> list[dict]:
    """Match exhibit references to cover page interactive data file / Inline XBRL."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r'\b104\b', text) and re.search(r'cover page|interactive data file|inline xbrl', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
