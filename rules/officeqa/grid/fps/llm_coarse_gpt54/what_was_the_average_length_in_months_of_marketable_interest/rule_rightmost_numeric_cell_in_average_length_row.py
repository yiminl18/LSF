def rule_rightmost_numeric_cell_in_average_length_row(doc: dict) -> list[dict]:
    """Return the rightmost numeric cell from rows mentioning average length in the target table."""
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
            for row, row_cells in by_row.items():
                row_text = " ".join((c.get("text") or "") for c in row_cells).lower()
                if "average length" in row_text:
                    nums = [c for c in row_cells if re.fullmatch(r'\d{1,3}', (c.get("text") or "").strip())]
                    if nums:
                        nums = sorted(nums, key=lambda x: (x.get("col", -1)))
                        out.append({"parent_span": span, "cell": nums[-1]})
    except Exception:
        return []
    return out
