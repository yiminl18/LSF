def rule_table_cells_withheld_other_refunds_net(doc: dict) -> list[dict]:
    """Match tables whose cells include the canonical Withheld/Other/Refunds/Net individual tax breakdown."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            texts = " ".join((c.get("text") or "") for c in cells)
            if all(re.search(p, texts, re.I) for p in [r'Withheld', r'Other', r'Refunds', r'\bNet\b']):
                out.append(span)
    except Exception:
        return []
    return out
