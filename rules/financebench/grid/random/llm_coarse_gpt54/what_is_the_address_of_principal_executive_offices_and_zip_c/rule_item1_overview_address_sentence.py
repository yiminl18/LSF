def rule_item1_overview_address_sentence(doc: dict) -> list[dict]:
    """Match Item 1/Overview text containing a full address sentence."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text") or ""
            if re.search(r'item 1|business|overview', path, re.I) and re.search(r'located at .*?\b\d{5}(?:-\d{4})?\b', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
