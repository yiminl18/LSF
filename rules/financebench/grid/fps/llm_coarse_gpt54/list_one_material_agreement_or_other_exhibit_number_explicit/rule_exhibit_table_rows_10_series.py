def rule_exhibit_table_rows_10_series(doc: dict) -> list[dict]:
    """Match exhibit tables with 10-series exhibit rows, often material agreements."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            first_col = [c for c in cells if c.get("col") == 0 and c.get("row", 0) > 0]
            if any(re.fullmatch(r'\s*10(\.\d+)?[A-Za-z]?\s*', (c.get("text", "") or "")) for c in first_col):
                out.append(span)
    except Exception:
        return []
    return out
