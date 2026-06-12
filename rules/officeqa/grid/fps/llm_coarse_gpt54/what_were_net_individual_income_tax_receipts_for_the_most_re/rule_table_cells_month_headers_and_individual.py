def rule_table_cells_month_headers_and_individual(doc: dict) -> list[dict]:
    """Match tables with month headers and an Individual income taxes row, common in quarter summary tables."""
    import re
    out = []
    try:
        month_patterns = [r'July', r'August', r'September', r'October|Oct', r'November|Nov', r'December|Dec']
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            texts = " ".join((c.get("text") or "") for c in cells)
            if re.search(r'Individual income taxes', texts, re.I):
                hits = sum(1 for pat in month_patterns if re.search(pat, texts, re.I))
                if hits >= 3:
                    out.append(span)
    except Exception:
        return []
    return out
