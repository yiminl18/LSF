def rule_table_cells_fiscal_to_date_row(doc: dict) -> list[dict]:
    """Match tables whose structured cells include a fiscal-to-date row label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(re.search(r'(Fiscal \d{4} to date|Actual fiscal year to date)', c.get("text") or "", re.I) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
