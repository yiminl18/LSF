def rule_revenue_row_top_of_table_under_item8(doc: dict) -> list[dict]:
    """Match Item 8 tables where the first or second data row is revenue/sales."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            if "item 8" not in path and "financial statements" not in path:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_num in sorted(rows)[:4]:
                row_join = " | ".join((c.get("text") or "") for c in rows[row_num]).lower()
                if any(k in row_join for k in ["net sales", "net revenues", "total revenue", "revenue", "sales"]):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
