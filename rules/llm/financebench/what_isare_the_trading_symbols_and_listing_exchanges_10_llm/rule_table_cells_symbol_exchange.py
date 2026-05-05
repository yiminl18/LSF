def rule_table_cells_symbol_exchange(doc: dict) -> list[dict]:
    """Return pseudo-spans from table cells containing symbol/exchange values in registration tables."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if not re.search(r"trading symbol|exchange|registered", joined, re.I):
                continue
            for c in cells:
                t = (c.get("text") or "").strip()
                if re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t) or re.search(r"nasdaq global select market|new york stock exchange|nyse", t, re.I):
                    out.append({
                        "text": t,
                        "label": "table_cell",
                        "page_no": span.get("page_no"),
                        "source_table_page_no": span.get("page_no"),
                        "table_cell": c,
                    })
        return out
    except Exception:
        return []
