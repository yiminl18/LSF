def rule_item1_business_principal_facilities_sentence(doc: dict) -> list[dict]:
    """Match Item 1/Business text mentioning principal facilities located at an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if re.search(r'Item 1|Business', path, re.I) and re.search(r'principal facilities are located at', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
