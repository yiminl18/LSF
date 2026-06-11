def rule_page1_table_multiple_rows_listing(doc: dict) -> list[dict]:
    """Match page-1 tables with multiple data rows under class/symbol/exchange headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table" or span.get("page_no") != 1:
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {c.get("row") for c in cells}
            txt = " | ".join((c.get("text", "") or "") for c in cells)
            if len(rows) >= 3 and re.search(r"trading symbol", txt, re.I) and re.search(r"exchange", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
