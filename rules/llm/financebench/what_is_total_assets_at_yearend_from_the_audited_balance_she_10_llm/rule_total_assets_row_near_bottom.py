def rule_total_assets_row_near_bottom(doc: dict) -> list[dict]:
    """Match tables where total assets appears in a later row, as is common in balance sheets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            td = span.get("table_data") or {}
            cells = td.get("cells") or []
            num_rows = td.get("num_rows") or 0
            if not num_rows:
                continue
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for r, row_cells in rows.items():
                row_text = " ".join((c.get("text") or "").lower() for c in row_cells)
                if "total assets" in row_text and r >= max(1, int(num_rows * 0.25)):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
