def rule_tables_with_assets_on_pages_after_item8(doc: dict) -> list[dict]:
    """Match asset tables on pages at or shortly after the Item 8 start page from the TOC."""
    try:
        item8_page = None
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (span.get("table_data") or {}).get("cells") or []
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " ".join((c.get("text") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0))).lower()
                if "item 8" in row_text and "financial statements" in row_text:
                    nums = [int((c.get("text") or "").strip()) for c in row_cells if (c.get("text") or "").strip().isdigit()]
                    if nums:
                        item8_page = nums[-1]
                        break
            if item8_page is not None:
                break
        out = []
        if item8_page is None:
            return out
        for span in doc.get("texts", []):
            if span.get("label") == "table" and item8_page <= int(span.get("page_no", 0)) <= item8_page + 10:
                txt = (span.get("text") or "").lower()
                if "assets" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
