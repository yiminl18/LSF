def rule_table_cells_end_of_period(doc: dict) -> list[dict]:
    """Match table spans containing end-of-period or reporting-date language near currency/coin terms."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'currency\s+and\s+coin', txt, re.I) and re.search(r'(end\s+of|reporting\s+date|date)', txt, re.I):
                out.append(span)
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_cc = any(re.search(r'currency|coin', (c.get("text") or ""), re.I) for c in cells)
            has_date = any(re.search(r'(end\s+of|date|latest|most\s+recent)', (c.get("text") or ""), re.I) for c in cells)
            if has_cc and has_date:
                out.append(span)
    except Exception:
        return []
    return out
