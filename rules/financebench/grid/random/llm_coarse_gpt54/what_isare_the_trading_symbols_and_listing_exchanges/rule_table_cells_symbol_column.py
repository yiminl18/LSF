def rule_table_cells_symbol_column(doc: dict) -> list[dict]:
    """Extract ticker-like cells from tables with a trading-symbol column."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            symbol_cols = {c.get("col") for c in cells if re.search(r"trading symbol", (c.get("text", "") or ""), re.I)}
            if not symbol_cols:
                continue
            for c in cells:
                if c.get("col") in symbol_cols and c.get("row", 0) > 0:
                    txt = (c.get("text", "") or "").strip()
                    if re.fullmatch(r"[A-Z]{1,6}(?:[/-][A-Z0-9]{1,6})?\d{0,2}", txt):
                        out.append(span)
                        break
        return out
    except Exception:
        return []
