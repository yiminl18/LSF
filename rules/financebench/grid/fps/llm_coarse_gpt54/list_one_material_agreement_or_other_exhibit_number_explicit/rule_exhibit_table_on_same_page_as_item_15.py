def rule_exhibit_table_on_same_page_as_item_15(doc: dict) -> list[dict]:
    """Match tables on pages where an Item 15 header appears."""
    import re
    out = []
    try:
        item_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                blob = ((span.get("text", "") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or ""))
                if re.search(r'item\s*15', blob, re.I):
                    item_pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in item_pages:
                out.append(span)
    except Exception:
        return []
    return out
