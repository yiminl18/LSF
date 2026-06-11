def rule_any_table_with_exhibit_like_first_column(doc: dict) -> list[dict]:
    """Match any table whose first column contains SEC exhibit-style numbers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            first_col_vals = [(c.get("text", "") or "") for c in cells if c.get("col") == 0]
            if any(re.fullmatch(r'\s*(2|4|10|99|104)(\.\d+)?[A-Za-z]?\s*', v) for v in first_col_vals):
                out.append(span)
    except Exception:
        return []
    return out
