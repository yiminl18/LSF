def rule_months_row_in_target_table(doc: dict) -> list[dict]:
    """Return rows/cells from the target table that mention months, since the answer is a month count."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells).lower()
            if "average length" not in joined:
                continue
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append(c)
            for row_cells in by_row.values():
                row_text = " ".join((c.get("text") or "") for c in row_cells).lower()
                if "month" in row_text:
                    out.append({"parent_span": span, "row_cells": row_cells})
    except Exception:
        return []
    return out
