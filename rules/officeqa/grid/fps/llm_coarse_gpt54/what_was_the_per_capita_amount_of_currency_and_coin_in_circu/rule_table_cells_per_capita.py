def rule_table_cells_per_capita(doc: dict) -> list[dict]:
    """Match table spans containing a cell with per capita wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                ctext = c.get("text") or ""
                if re.search(r'per\s+capita', ctext, re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
