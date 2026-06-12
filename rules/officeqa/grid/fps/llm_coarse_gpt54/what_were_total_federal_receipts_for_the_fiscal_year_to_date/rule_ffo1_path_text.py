def rule_ffo1_path_text(doc: dict) -> list[dict]:
    """Match spans under a path_text containing Table FFO-1 / Summary of Fiscal Operations."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'(Table\s*)?FFO[-\s]?1', path, re.I) or re.search(r'Summary of Fiscal Operations', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
