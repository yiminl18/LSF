def rule_table_cells_private_investors(doc: dict) -> list[dict]:
    """Match table spans whose cells mention private investors together with average length."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells).lower()
            if "private investors" in joined and "average length" in joined:
                out.append(span)
    except Exception:
        return []
    return out
