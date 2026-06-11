def rule_exhibit_table_rows_99_or_104(doc: dict) -> list[dict]:
    """Match exhibit tables with 99/104 rows for press release or cover page answers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            first_col = [c for c in cells if c.get("col") == 0 and c.get("row", 0) > 0]
            if any(re.fullmatch(r'\s*(99(\.\d+)?|104)\s*', (c.get("text", "") or "")) for c in first_col):
                out.append(span)
    except Exception:
        return []
    return out
