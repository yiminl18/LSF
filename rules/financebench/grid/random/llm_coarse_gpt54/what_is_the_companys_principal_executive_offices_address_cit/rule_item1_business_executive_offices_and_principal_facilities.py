def rule_item1_business_executive_offices_and_principal_facilities(doc: dict) -> list[dict]:
    """Match Item 1/Business text stating executive offices and principal facilities are located at an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if re.search(r'Item 1|Business', path, re.I) and re.search(r'executive offices and principal facilities are located at', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
