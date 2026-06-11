def rule_exhibit_index_near_item_15_toc_page(doc: dict) -> list[dict]:
    """Match TOC rows that mention Exhibit Index or Item 15 page references."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if re.search(r"item\s*15", txt, re.I) and re.search(r"exhibit index|exhibits?", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
