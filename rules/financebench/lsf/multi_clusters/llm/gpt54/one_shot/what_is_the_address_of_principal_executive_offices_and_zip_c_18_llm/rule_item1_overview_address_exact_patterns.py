def rule_item1_overview_address_exact_patterns(doc: dict) -> list[dict]:
    """Match Item 1/Overview spans containing known prose address formulations."""
    import re
    try:
        pats = [
            r'Our executive offices and principal facilities are located at',
            r'Our principal executive offices are located at',
            r'Our principal corporate offices are located in',
            r'Our executive offices .* located at',
        ]
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if re.search(r'item 1|business|overview', path, re.I) and any(re.search(p, text, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
