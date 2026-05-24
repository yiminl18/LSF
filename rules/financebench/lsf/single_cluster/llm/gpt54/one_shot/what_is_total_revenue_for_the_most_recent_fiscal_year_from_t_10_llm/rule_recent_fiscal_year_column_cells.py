def rule_recent_fiscal_year_column_cells(doc: dict) -> list[dict]:
    """Match numeric cells in financial tables that sit under the most recent year column."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            if not cells:
                continue
            years = []
            for c in cells:
                txt = (c.get("text") or "").strip()
                if re.fullmatch(r"(20\d{2}|19\d{2})", txt):
                    years.append((int(txt), c.get("col")))
            if not years:
                continue
            recent_year, recent_col = sorted(years)[-1]
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c)
            for row, row_cells in row_texts.items():
                row_join = " | ".join((c.get("text") or "") for c in row_cells).lower()
                if any(k in row_join for k in ["net sales", "net revenues", "total revenue", "revenue", "sales"]):
                    for c in row_cells:
                        if c.get("col") == recent_col:
                            out.append(span)
                            break
                    break
        return out
    except Exception:
        return []
