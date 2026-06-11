def rule_path_contains_item_15_exhibits(doc: dict) -> list[dict]:
    """Match spans under Item 15 exhibits sections via structure.path_text."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*15", path, re.I) and re.search(r"exhibit", path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
