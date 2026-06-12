def rule_page_with_table_ffo1_heading(doc: dict) -> list[dict]:
    """Match all table spans on pages containing a heading 'Table FFO-1'."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            if re.search(r'table\s+FFO[-\s]?1', span.get("text", ""), re.I):
                pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
