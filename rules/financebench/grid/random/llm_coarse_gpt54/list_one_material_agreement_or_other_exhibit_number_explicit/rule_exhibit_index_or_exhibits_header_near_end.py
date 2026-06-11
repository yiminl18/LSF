def rule_exhibit_index_or_exhibits_header_near_end(doc: dict) -> list[dict]:
    """Match section headers near the end that mention Exhibit Index or Exhibits."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        max_page = max((s.get("page_no") or 0) for s in texts) if texts else 0
        for span in texts:
            if span.get("label") != "section_header":
                continue
            page = span.get("page_no") or 0
            txt = span.get("text") or ""
            if max_page and page >= max_page - 8 and re.search(r"\b(exhibit index|exhibits?)\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
