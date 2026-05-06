def rule_tables_on_item8_toc_pages(doc: dict) -> list[dict]:
    """Match tables on pages referenced by the table of contents for Item 8, if those pages are present in extracted spans."""
    try:
        texts = doc.get("texts", [])
        item8_pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            txt = (span.get("text") or "") + "\n" + path
            if "table of contents" not in txt.lower() and "index" not in txt.lower():
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_join = " | ".join((c.get("text") or "") for c in row_cells).lower()
                if "item 8" in row_join and "financial statements" in row_join:
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if t.isdigit():
                            item8_pages.add(int(t))
        out = []
        if not item8_pages:
            return out
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") in item8_pages:
                out.append(span)
        return out
    except Exception:
        return []
