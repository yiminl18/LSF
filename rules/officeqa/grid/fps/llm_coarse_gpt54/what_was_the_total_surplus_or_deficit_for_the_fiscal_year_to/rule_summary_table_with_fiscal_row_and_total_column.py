def rule_summary_table_with_fiscal_row_and_total_column(doc: dict) -> list[dict]:
    """Match tables whose cells include both a fiscal-to-date row label and a total-surplus/deficit header."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_row = False
            has_col = False
            for c in cells:
                t = c.get("text", "")
                if re.search(r'fiscal\s+(year|\d{4})\s+to\s+date', t, re.I):
                    has_row = True
                if re.search(r'total surplus.*deficit', t, re.I):
                    has_col = True
            if has_row and has_col:
                out.append(span)
    except Exception:
        return []
    return out
