def rule_table_cells_incorporation_or_irs(doc: dict) -> list[dict]:
    """Match table spans whose cells mention incorporation/jurisdiction or IRS identification."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                cells = (((span.get("table_data") or {}).get("cells")) or [])
                joined = " ".join((c.get("text", "") or "") for c in cells)
                if re.search(r"state or other jurisdiction of incorporation|i\.?r\.?s\.? employer identification|employer identification no", joined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
