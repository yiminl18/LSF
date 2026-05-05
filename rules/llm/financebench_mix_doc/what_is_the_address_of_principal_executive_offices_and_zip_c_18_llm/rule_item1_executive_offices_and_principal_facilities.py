def rule_item1_executive_offices_and_principal_facilities(doc: dict) -> list[dict]:
    """Match Item 1/Business spans with the phrase executive offices and principal facilities."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if re.search(r'item 1|business|overview', path, re.I) and re.search(r'executive offices and principal facilities', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
