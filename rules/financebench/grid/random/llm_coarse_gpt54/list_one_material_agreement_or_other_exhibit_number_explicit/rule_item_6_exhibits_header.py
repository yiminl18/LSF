def rule_item_6_exhibits_header(doc: dict) -> list[dict]:
    """Match 10-Q Item 6 Exhibits headers."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*6", txt + " " + path, re.I) and re.search(r"\bexhibits?\b", txt + " " + path, re.I):
                out.append(span)
    except Exception:
        return []
    return out
