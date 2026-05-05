def rule_row_header_revenue_generic(doc: dict) -> list[dict]:
    """Match tables with a first-column row mentioning revenue or sales near the top of the table."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                row = c.get("row", 999)
                col = c.get("col", 999)
                txt = (c.get("text") or "").strip().lower()
                if row <= 5 and col <= 1 and any(k in txt for k in ["revenue", "revenues", "sales"]):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
