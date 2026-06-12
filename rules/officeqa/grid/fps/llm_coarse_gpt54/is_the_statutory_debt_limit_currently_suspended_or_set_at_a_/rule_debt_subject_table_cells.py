def rule_debt_subject_table_cells(doc: dict) -> list[dict]:
    """Match table spans whose cell texts mention debt subject to statutory limit/limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if re.search(r"debt subject to statutory (limit|limitation)|statutory (limit|limitation)", joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
