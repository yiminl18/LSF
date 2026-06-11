def rule_exhibit_index_path_or_text(doc: dict) -> list[dict]:
    """Match any span under an Exhibit Index path or with Exhibit Index text."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r'exhibit index', path + " " + text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
