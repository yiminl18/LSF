def rule_esf_total_assets_cell_row(doc: dict) -> list[dict]:
    """Match table spans having a cell exactly or nearly equal to Total assets."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                t = (c.get("text", "") or "").strip()
                if re.fullmatch(r'total assets\.?', t, re.I) or re.search(r'^\s*total assets\b', t, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
