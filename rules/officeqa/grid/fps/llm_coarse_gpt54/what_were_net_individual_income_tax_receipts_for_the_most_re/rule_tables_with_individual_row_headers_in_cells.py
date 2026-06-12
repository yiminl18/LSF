def rule_tables_with_individual_row_headers_in_cells(doc: dict) -> list[dict]:
    """Match tables whose parsed cells include row/header text for Individual income taxes."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                t = c.get("text") or ""
                if re.search(r'Individual income taxes|Individual', t, re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
