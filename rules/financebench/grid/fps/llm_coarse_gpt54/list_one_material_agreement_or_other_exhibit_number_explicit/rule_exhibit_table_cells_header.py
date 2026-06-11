def rule_exhibit_table_cells_header(doc: dict) -> list[dict]:
    """Match tables whose cell headers include Exhibit Number/Description."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            headers = " | ".join((c.get("text", "") or "") for c in cells if c.get("is_column_header"))
            if re.search(r'exhibit', headers, re.I) and re.search(r'description', headers, re.I):
                out.append(span)
    except Exception:
        return []
    return out
