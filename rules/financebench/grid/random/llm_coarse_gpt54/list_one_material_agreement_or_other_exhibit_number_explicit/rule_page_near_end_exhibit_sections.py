def rule_page_near_end_exhibit_sections(doc: dict) -> list[dict]:
    """Match spans on late pages mentioning exhibits, since exhibit indexes usually appear near the end."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        max_page = max((s.get("page_no") or 0) for s in texts) if texts else 0
        for span in texts:
            page = span.get("page_no") or 0
            txt = span.get("text") or ""
            if max_page and page >= max_page - 5 and re.search(r"\bexhibit", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
