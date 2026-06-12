def rule_table_cells_total_receipts_row(doc: dict) -> list[dict]:
    """Match tables whose structured cells include a Total receipts row label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(re.search(r'(Total receipts|Net receipts|Net budget receipts)', c.get("text") or "", re.I) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
