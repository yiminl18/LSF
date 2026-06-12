def rule_table_cells_individual_net_column(doc: dict) -> list[dict]:
    """Match tables whose cell structure includes an Individual/Net header combination."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            texts = " ".join((c.get("text") or "") for c in cells)
            if re.search(r'Individual', texts, re.I) and re.search(r'\bNet\b', texts, re.I):
                out.append(span)
    except Exception:
        return []
    return out
