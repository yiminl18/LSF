def rule_tables_with_total_assets_and_consolidated_balance_sheet_toc_reference(doc: dict) -> list[dict]:
    """Match tables when the document TOC references a consolidated balance sheet page and the table is on/near that page."""
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
                row_text = " ".join((c.get("text") or "").lower() for c in row_cells)
                if "consolidated balance sheet" in row_text or "balance sheet" in row_text:
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if t.isdigit():
                            target_pages.add(int(t))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            p = int(span.get("page_no", -999))
            if any(tp <= p <= tp + 1 for tp in target_pages):
                txt = (span.get("text") or "").lower()
                if "assets" in txt or "total assets" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
