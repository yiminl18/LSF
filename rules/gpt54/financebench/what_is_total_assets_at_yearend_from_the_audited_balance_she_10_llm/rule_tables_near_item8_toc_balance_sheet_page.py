def rule_tables_near_item8_toc_balance_sheet_page(doc: dict) -> list[dict]:
    """Use the table of contents row for balance sheet to find tables on that page."""
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " | ".join((c.get("text") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0))).lower()
                if "balance sheet" in row_text or "statement of financial position" in row_text:
                    for c in row_cells:
                        txt = (c.get("text") or "").strip()
                        if txt.isdigit():
                            target_pages.add(int(txt))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in target_pages:
                out.append(span)
        return out
    except Exception:
        return []
