def rule_tables_with_total_assets_in_row_header(doc: dict) -> list[dict]:
    """Match tables where total assets is explicitly marked as a row header."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(c.get("is_row_header") and "total assets" in (c.get("text") or "").lower() for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
