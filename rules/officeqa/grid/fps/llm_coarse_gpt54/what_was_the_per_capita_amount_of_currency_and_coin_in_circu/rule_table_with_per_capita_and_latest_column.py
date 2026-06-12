def rule_table_with_per_capita_and_latest_column(doc: dict) -> list[dict]:
    """Match tables likely containing the answer by requiring both per capita and latest/current date cues."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'per\s+capita', txt, re.I) and re.search(r'(latest|most\s+recent|current|date)', txt, re.I):
                out.append(span)
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_pc = any(re.search(r'per\s+capita', (c.get("text") or ""), re.I) for c in cells)
            has_recent = any(re.search(r'(latest|most\s+recent|current|date)', (c.get("text") or ""), re.I) for c in cells)
            if has_pc and has_recent:
                out.append(span)
    except Exception:
        return []
    return out
