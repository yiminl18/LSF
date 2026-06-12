def rule_ffo1_summary_tables(doc: dict) -> list[dict]:
    """Match table spans for Table FFO-1 / Summary of Fiscal Operations, where the debt answer often appears."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if (
                re.search(r'ffo[\-–— ]?1', path, re.I)
                or re.search(r'summary of fiscal operations', path, re.I)
                or re.search(r'ffo[\-–— ]?1', text, re.I)
                or re.search(r'summary of fiscal operations', text, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
