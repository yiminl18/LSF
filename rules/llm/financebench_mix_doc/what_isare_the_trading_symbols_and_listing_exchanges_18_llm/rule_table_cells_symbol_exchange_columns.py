def rule_table_cells_symbol_exchange_columns(doc: dict) -> list[dict]:
    """Match table spans whose cells include trading symbol and exchange column headers."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_texts = " | ".join((c.get("text") or "").lower() for c in cells)
            if (
                "trading symbol" in cell_texts
                and "exchange" in cell_texts
            ) or (
                "name of each exchange on which registered" in cell_texts
            ):
                out.append(span)
        return out
    except Exception:
        return []
