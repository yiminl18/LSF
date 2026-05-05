def rule_tables_with_total_assets_and_multiple_numeric_cells(doc: dict) -> list[dict]:
    """Match tables where the total assets row has multiple numeric cells, indicating year-end values."""
    import re
    try:
        num_re = re.compile(r"^\$?\(?\d[\d,\.]*\)?$")
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_sorted = sorted(row_cells, key=lambda x: x.get("col", 0))
                row_texts = [(c.get("text") or "").strip() for c in row_sorted]
                if any("total assets" in t.lower() for t in row_texts):
                    numeric_count = sum(1 for t in row_texts if num_re.match(t.replace(" ", "")))
                    if numeric_count >= 1:
                        out.append(span)
                        break
        return out
    except Exception:
        return []
