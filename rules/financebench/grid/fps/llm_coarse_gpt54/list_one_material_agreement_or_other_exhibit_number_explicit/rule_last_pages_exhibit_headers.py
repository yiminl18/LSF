def rule_last_pages_exhibit_headers(doc: dict) -> list[dict]:
    """Match exhibit-related headers on the last few pages."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []
        max_page = max((s.get("page_no") or 0) for s in texts)
        for span in texts:
            if (span.get("page_no") or 0) >= max_page - 3 and span.get("label") == "section_header":
                text = span.get("text", "") or ""
                path = (span.get("structure", {}) or {}).get("path_text", "") or ""
                if re.search(r'exhibit|financial statements and exhibits|item\s*9\.01|item\s*15', text + " " + path, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
