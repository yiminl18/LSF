def rule_ffo1_heading_path(doc: dict) -> list[dict]:
    """Match spans under a path mentioning Table FFO-1 / Summary of Fiscal Operations."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'(table\s+)?FFO[-\s]?1', path, re.I) or re.search(r'summary of fiscal operations', path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
