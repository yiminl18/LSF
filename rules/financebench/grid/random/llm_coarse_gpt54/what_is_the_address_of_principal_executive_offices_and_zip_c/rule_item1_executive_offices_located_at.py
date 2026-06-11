def rule_item1_executive_offices_located_at(doc: dict) -> list[dict]:
    """Match Item 1 business text stating executive offices are located at an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'executive offices.*located at', txt, re.I) and re.search(r'item 1|business', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
