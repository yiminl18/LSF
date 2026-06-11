def rule_short_8k_item_901_text(doc: dict) -> list[dict]:
    """Match short 8-K exhibit listing text under Item 9.01 on page 2-4."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no") or 0
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if 2 <= page <= 4 and re.search(r'item\s*9\.01', path + " " + text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
