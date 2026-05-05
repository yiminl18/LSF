def rule_tables_with_total_assets_in_first_column(doc: dict) -> list[dict]:
    """Match tables where total assets appears in the first/leftmost column, typical of row labels."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any((c.get("col") == 0 and "total assets" in (c.get("text") or "").lower()) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
