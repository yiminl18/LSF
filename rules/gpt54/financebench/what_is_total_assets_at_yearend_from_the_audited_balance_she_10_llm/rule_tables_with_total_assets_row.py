def rule_tables_with_total_assets_row(doc: dict) -> list[dict]:
    """Match any table containing a row/cell mentioning total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any("total assets" in (c.get("text") or "").lower() for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
