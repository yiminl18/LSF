def rule_long_term_debt_table_cell_rows(doc: dict) -> list[dict]:
    """Match tables whose cells include a long-term debt row and at least one adjacent numeric cell."""
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
            matched = False
            for r, row_cells in by_row.items():
                row_labels = " ".join((c.get("text") or "") for c in row_cells).lower()
                if re.search(r"long[- ]term debt|debt.*less current portion", row_labels):
                    nums = [c for c in row_cells if re.search(r"\$?\s*\(?\d[\d,]*(\.\d+)?\)?", c.get("text") or "")]
                    if nums:
                        matched = True
                        break
            if matched:
                out.append(span)
    except Exception:
        return []
    return out
