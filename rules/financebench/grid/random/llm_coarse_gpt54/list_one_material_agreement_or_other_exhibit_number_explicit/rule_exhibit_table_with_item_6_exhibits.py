def rule_exhibit_table_with_item_6_exhibits(doc: dict) -> list[dict]:
    """Match 10-Q exhibit tables under Item 6 Exhibits."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text") or ""
            if span.get("label") == "table" and (
                re.search(r"item\s*6", path, re.I) or re.search(r"item\s*6", txt, re.I)
            ) and re.search(r"\bexhibits?\b", path + " " + txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
