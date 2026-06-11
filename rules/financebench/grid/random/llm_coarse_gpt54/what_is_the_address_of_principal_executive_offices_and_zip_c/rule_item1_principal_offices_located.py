def rule_item1_principal_offices_located(doc: dict) -> list[dict]:
    """Match Item 1 business text stating principal offices are located in a city/state."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'principal (corporate )?offices are located', txt, re.I) and re.search(r'item 1|business', path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
