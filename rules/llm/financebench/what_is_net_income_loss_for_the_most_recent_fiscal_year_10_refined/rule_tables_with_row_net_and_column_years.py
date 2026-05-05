def rule_tables_with_row_net_and_column_years(doc: dict) -> list[dict]:
    """Match tables where a net row coexists with year-like column headers."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            cells = ((s.get("table_data") or {}).get("cells") or [])
            has_net = any(re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", c.get("text", "") or "", re.I) for c in cells)
            has_year_header = any(c.get("is_column_header") and re.search(r"\b20\d{2}\b", c.get("text", "") or "") for c in cells)
            if has_net and has_year_header:
                out.append(s)
        return out
    except Exception:
        return []
