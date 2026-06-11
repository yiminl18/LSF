def rule_exhibit_index_header_span(doc: dict) -> list[dict]:
    """Match Exhibit Index headers or paths."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                text = span.get("text", "") or ""
                path = (span.get("structure", {}) or {}).get("path_text", "") or ""
                if re.search(r'exhibit index', text, re.I) or re.search(r'exhibit index', path, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
