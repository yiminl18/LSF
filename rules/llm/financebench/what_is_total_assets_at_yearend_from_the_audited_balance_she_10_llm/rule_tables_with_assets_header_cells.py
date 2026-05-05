def rule_tables_with_assets_header_cells(doc: dict) -> list[dict]:
    """Match tables whose cells include assets-related headers."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            header_hits = 0
            for c in cells:
                txt = (c.get("text") or "").lower()
                if c.get("is_row_header") or c.get("is_column_header"):
                    if "assets" in txt or "current assets" in txt or "total assets" in txt:
                        header_hits += 1
            if header_hits >= 1:
                out.append(span)
        return out
    except Exception:
        return []
