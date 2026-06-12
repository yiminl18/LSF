def rule_table_cells_receipts_and_fytd(doc: dict) -> list[dict]:
    """Match tables whose cells include both a receipts label and a fiscal-to-date label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_receipts = any(re.search(r'(Total receipts|Net receipts|Net budget receipts)', c.get("text") or "", re.I) for c in cells)
            has_fytd = any(re.search(r'(Fiscal \d{4} to date|Actual fiscal year to date)', c.get("text") or "", re.I) for c in cells)
            if has_receipts and has_fytd:
                out.append(span)
        return out
    except Exception:
        return []
