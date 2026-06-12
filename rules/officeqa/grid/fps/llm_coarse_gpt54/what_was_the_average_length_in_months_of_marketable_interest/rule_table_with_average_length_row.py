def rule_table_with_average_length_row(doc: dict) -> list[dict]:
    """Match table spans that likely contain a dedicated row/header for average length in months."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            texts = [(c.get("text") or "").lower() for c in cells]
            if any("average length" in t for t in texts) and any("month" in t for t in texts):
                out.append(span)
    except Exception:
        return []
    return out
