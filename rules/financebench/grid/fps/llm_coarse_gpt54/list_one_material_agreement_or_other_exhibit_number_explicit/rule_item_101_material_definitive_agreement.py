def rule_item_101_material_definitive_agreement(doc: dict) -> list[dict]:
    """Match Item 1.01 sections that often name the agreement later listed in exhibits."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r'item\s*1\.01', path + " " + text, re.I) and re.search(r'agreement|indenture|amendment', path + " " + text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
