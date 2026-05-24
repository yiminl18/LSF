def rule_sales_keyword_in_table_cells(doc: dict) -> list[dict]:
    """Match tables containing any cell with sales/net sales text."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            if any(("sales" in (c.get("text") or "").lower()) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
