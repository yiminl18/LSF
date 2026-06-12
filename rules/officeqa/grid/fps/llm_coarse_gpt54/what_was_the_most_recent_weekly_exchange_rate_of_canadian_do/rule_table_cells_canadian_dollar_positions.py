def rule_table_cells_canadian_dollar_positions(doc: dict) -> list[dict]:
    """Match table spans whose parsed cells mention Canadian dollar positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for cell in cells:
                ctext = (cell.get("text") or "").lower()
                if "canadian dollar positions" in ctext:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
