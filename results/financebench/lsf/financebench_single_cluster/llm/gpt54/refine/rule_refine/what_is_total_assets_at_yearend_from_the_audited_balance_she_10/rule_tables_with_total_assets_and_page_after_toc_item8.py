def rule_tables_with_total_assets_and_page_after_toc_item8(doc: dict) -> list[dict]:
    """Match tables on the exact or next page after the TOC's Item 8 page number."""
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
                row_text = " ".join((c.get("text") or "").lower() for c in row_cells)
                if "item 8" in row_text:
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if t.isdigit():
                            item8_pages.add(int(t))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                p = int(span.get("page_no", -999))
                if any(p in {ip, ip + 1} for ip in item8_pages):
                    txt = (span.get("text") or "").lower()
                    if "assets" in txt or "balance sheet" in txt or "financial position" in txt:
                        out.append(span)
        return out
    except Exception:
        return []
