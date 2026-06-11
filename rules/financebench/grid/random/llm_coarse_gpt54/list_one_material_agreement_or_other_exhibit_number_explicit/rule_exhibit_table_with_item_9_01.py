def rule_exhibit_table_with_item_9_01(doc: dict) -> list[dict]:
    """Match 8-K exhibit tables under Item 9.01 Financial Statements and Exhibits."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text") or ""
            if span.get("label") == "table" and (
                re.search(r"item\s*9\.?01", path, re.I) or re.search(r"item\s*9\.?01", txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
