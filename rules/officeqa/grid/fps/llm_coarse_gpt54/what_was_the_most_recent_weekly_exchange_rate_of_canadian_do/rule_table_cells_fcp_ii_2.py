def rule_table_cells_fcp_ii_2(doc: dict) -> list[dict]:
    """Match table spans whose parsed cells contain FCP-II-2."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for cell in cells:
                ctext = (cell.get("text") or "").lower()
                if "fcp-ii-2" in ctext or "fcp-1i-2" in ctext:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
