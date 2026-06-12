def rule_table_cells_most_recent_date(doc: dict) -> list[dict]:
    """Match table spans containing a most recent/latest date column or row, where the answer is likely taken."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            hit_recent = False
            hit_percap = False
            for c in cells:
                ctext = c.get("text") or ""
                if re.search(r'(latest|most\s+recent|current)\b', ctext, re.I):
                    hit_recent = True
                if re.search(r'per\s+capita', ctext, re.I):
                    hit_percap = True
            if hit_recent or hit_percap:
                out.append(span)
    except Exception:
        return []
    return out
