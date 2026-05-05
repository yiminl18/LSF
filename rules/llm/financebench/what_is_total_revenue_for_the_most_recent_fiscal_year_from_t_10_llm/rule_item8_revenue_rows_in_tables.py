def rule_item8_revenue_rows_in_tables(doc: dict) -> list[dict]:
    """Match Item 8 tables containing revenue-like row labels such as net sales, net revenues, or revenue."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Item 8" not in path and "Financial Statements" not in path and "Supplementary Data" not in path:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row = c.get("row")
                row_texts.setdefault(row, []).append((c.get("text") or "").strip())
            found = False
            for parts in row_texts.values():
                row_join = " | ".join(parts).lower()
                if any(k in row_join for k in [
                    "net sales", "net revenues", "total revenue", "total revenues",
                    "revenue", "revenues", "sales"
                ]):
                    found = True
                    break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
