def rule_exhibit_with_cover_page_interactive_data_file(doc: dict) -> list[dict]:
    """Match spans containing an exhibit number near 'Cover Page Interactive Data File'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I) and re.search(r"cover page interactive data file", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
