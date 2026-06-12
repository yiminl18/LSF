def rule_table_cells_population_and_per_capita(doc: dict) -> list[dict]:
    """Match table spans containing both population and per capita concepts."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'population', txt, re.I) and re.search(r'per\s+capita', txt, re.I):
                out.append(span)
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_pop = any(re.search(r'population', (c.get("text") or ""), re.I) for c in cells)
            has_pc = any(re.search(r'per\s+capita', (c.get("text") or ""), re.I) for c in cells)
            if has_pop and has_pc:
                out.append(span)
    except Exception:
        return []
    return out
