def rule_total_assets_row_with_numeric_values(doc: dict) -> list[dict]:
    """Match tables where the total assets row also contains numeric values."""
    import re
    try:
        num_re = re.compile(r"[\$\(]?\d[\d,\.]*")
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " | ".join((c.get("text") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0)))
                if "total assets" in row_text.lower() and len(num_re.findall(row_text)) >= 1:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
