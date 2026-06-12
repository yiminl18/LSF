def rule_table_cells_average_length_phrase(doc: dict) -> list[dict]:
    """Match table spans whose cells contain the phrase 'average length'."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells).lower()
            if "average length" in joined and "marketable" in joined:
                out.append(span)
    except Exception:
        return []
    return out
