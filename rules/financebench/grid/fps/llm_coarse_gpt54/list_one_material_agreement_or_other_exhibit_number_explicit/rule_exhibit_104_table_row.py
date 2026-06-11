def rule_exhibit_104_table_row(doc: dict) -> list[dict]:
    """Match exhibit tables containing a 104 row with cover page description."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for r, vals in rows.items():
                joined = " | ".join(vals)
                if re.search(r'\b104\b', joined) and re.search(r'cover page|interactive data file|inline xbrl', joined, re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
