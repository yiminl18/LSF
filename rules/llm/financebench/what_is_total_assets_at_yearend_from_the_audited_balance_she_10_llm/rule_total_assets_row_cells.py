def rule_total_assets_row_cells(doc: dict) -> list[dict]:
    """Return synthetic row-like matches as the table span when a table has a row header/cell equal to total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_idx, row_cells in rows.items():
                row_texts = [(c.get("text") or "").strip().lower() for c in row_cells]
                if any(t == "total assets" or t.startswith("total assets") for t in row_texts):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
