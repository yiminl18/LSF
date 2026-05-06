def rule_tables_on_item8_pages(doc: dict) -> list[dict]:
    """Match tables on or near pages referenced by Item 8 in the table of contents."""
    try:
        item8_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " | ".join((c.get("text") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0))).lower()
                if "item 8" in row_text and "financial statements" in row_text:
                    for c in row_cells:
                        txt = (c.get("text") or "").strip()
                        if txt.isdigit():
                            item8_pages.add(int(txt))
        out = []
        if not item8_pages:
            return out
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in item8_pages.union({p + 1 for p in item8_pages}, {p + 2 for p in item8_pages}):
                out.append(span)
        return out
    except Exception:
        return []
