def rule_table_rows_with_average_length_and_numeric(doc: dict) -> list[dict]:
    """Return row groups from tables where row text mentions average length and contains a numeric value."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append(c)
            for row_cells in by_row.values():
                row_text = " ".join((c.get("text") or "") for c in row_cells).lower()
                if "average length" in row_text and any(re.fullmatch(r'\d{1,3}', (c.get("text") or "").strip()) for c in row_cells):
                    out.append({"parent_span": span, "row_cells": row_cells})
    except Exception:
        return []
    return out
