def rule_numeric_cells_in_revenue_rows(doc: dict) -> list[dict]:
    """Match tables where a revenue/sales row has numeric values in later columns."""
    import re
    try:
        out = []
        num_re = re.compile(r"^\$?\(?-?[\d,]+(?:\.\d+)?\)?$")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            found = False
            for row_cells in rows.values():
                row_join = " | ".join((c.get("text") or "") for c in row_cells).lower()
                if any(k in row_join for k in ["net sales", "net revenues", "total revenue", "revenue", "sales"]):
                    numeric_count = 0
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if num_re.match(t):
                            numeric_count += 1
                    if numeric_count >= 1:
                        found = True
                        break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
