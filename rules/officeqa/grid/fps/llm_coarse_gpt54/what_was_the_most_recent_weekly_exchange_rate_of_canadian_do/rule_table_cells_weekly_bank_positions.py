def rule_table_cells_weekly_bank_positions(doc: dict) -> list[dict]:
    """Match table spans whose parsed cells contain Weekly Bank Positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for cell in cells:
                ctext = (cell.get("text") or "").lower()
                if "weekly bank positions" in ctext:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
