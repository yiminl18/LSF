def rule_top_row_revenue_in_income_tables(doc: dict) -> list[dict]:
    """Match income-statement tables where the revenue row appears in the first few rows."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = ((span.get("text") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or "")).lower()
            if not any(k in txt for k in ["income", "operations", "earnings"]):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_num, row_cells in rows.items():
                if row_num is None or row_num > 4:
                    continue
                row_join = " | ".join((c.get("text") or "") for c in row_cells).lower()
                if any(k in row_join for k in ["net sales", "net revenues", "total revenue", "revenue", "sales"]):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
