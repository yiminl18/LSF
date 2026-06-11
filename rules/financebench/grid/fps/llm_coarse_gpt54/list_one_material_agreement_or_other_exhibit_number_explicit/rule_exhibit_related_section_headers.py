def rule_exhibit_related_section_headers(doc: dict) -> list[dict]:
    """Match section headers containing exhibit-related words or item numbers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            blob = (span.get("text", "") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or "")
            if re.search(r'exhibit|financial statements and exhibits|item\s*9\.01|item\s*15|exhibit index', blob, re.I):
                out.append(span)
    except Exception:
        return []
    return out
