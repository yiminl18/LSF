def rule_numeric_cells_in_target_table(doc: dict) -> list[dict]:
    """Return numeric-looking cells from the target average-length table."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells).lower()
            if not ("average length" in joined and "marketable" in joined):
                continue
            for c in cells:
                t = (c.get("text") or "").strip()
                if re.fullmatch(r'\d{1,3}', t):
                    out.append({"parent_span": span, "cell": c})
    except Exception:
        return []
    return out
