def rule_exhibit_index_after_item_15_page(doc: dict) -> list[dict]:
    """Use TOC Item 15 page number and return exhibit-related spans on pages at or after that page."""
    import re
    try:
        texts = doc.get("texts", [])
        min_page = None
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " ".join((c.get("text") or "") for c in row_cells)
                if re.search(r"\bitem\s*15\b", row_text, re.I) and re.search(r"\bexhibit", row_text, re.I):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if re.fullmatch(r"\d{1,4}", t):
                            p = int(t)
                            min_page = p if min_page is None else min(min_page, p)
        if min_page is None:
            return []
        out = []
        for span in texts:
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if span.get("page_no", 0) >= min_page and ("exhibit" in txt.lower() or "exhibit" in path.lower()):
                out.append(span)
        return out
    except Exception:
        return []
