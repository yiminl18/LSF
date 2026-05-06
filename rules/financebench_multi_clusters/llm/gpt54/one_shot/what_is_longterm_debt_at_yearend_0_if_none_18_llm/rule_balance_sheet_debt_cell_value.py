def rule_balance_sheet_debt_cell_value(doc: dict) -> list[dict]:
    """Return balance sheet table spans where a debt row has a numeric value cell."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append(c)
            found = False
            for r, row_cells in by_row.items():
                row_join = " ".join((c.get("text") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0))).lower()
                if re.search(r"long[- ]term debt|debt.*less current portion", row_join):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if re.search(r"\$?\s*\(?\d[\d,]*(\.\d+)?\)?", t):
                            found = True
                            break
                if found:
                    break
            if found:
                out.append(span)
    except Exception:
        return []
    return out
