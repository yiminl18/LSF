def rule_fiscal_summary_table_by_path_and_label(doc: dict) -> list[dict]:
    """Match table spans whose path_text itself is the FFO-1 summary section."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'(table\s+)?FFO[-\s]?1', path, re.I) or re.search(r'summary of fiscal operations', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
