def rule_exhibit_description_column_rows(doc: dict) -> list[dict]:
    """Match exhibit tables by detecting first column exhibit numbers and second column descriptions."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), {})[c.get("col")] = c.get("text", "") or ""
            hits = 0
            for r, cols in by_row.items():
                if r == 0:
                    continue
                c0 = cols.get(0, "")
                c1 = cols.get(1, "")
                if re.fullmatch(r'\s*(2|4|10|99|104)(\.\d+)?[A-Za-z]?\s*', c0) and len(c1) > 5:
                    hits += 1
            if hits >= 1:
                out.append(span)
    except Exception:
        return []
    return out
