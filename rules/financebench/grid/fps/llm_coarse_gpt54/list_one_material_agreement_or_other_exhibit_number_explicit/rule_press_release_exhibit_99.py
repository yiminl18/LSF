def rule_press_release_exhibit_99(doc: dict) -> list[dict]:
    """Match exhibit references to press releases, often Exhibit 99.1."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r'99(\.\d+)?', text) and re.search(r'press release', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
