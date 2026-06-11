def rule_table_cells_exchange_column(doc: dict) -> list[dict]:
    """Extract exchange-name cells from tables with an exchange column."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            exch_cols = {
                c.get("col")
                for c in cells
                if re.search(r"name of each exchange on which registered", (c.get("text", "") or ""), re.I)
                or re.search(r"name of exchange on which registered", (c.get("text", "") or ""), re.I)
            }
            if not exch_cols:
                continue
            for c in cells:
                if c.get("col") in exch_cols and c.get("row", 0) > 0:
                    txt = c.get("text", "") or ""
                    if re.search(r"exchange|nasdaq", txt, re.I):
                        out.append(span)
                        break
        return out
    except Exception:
        return []
