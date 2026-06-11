def rule_exhibit_3_or_4_family(doc: dict) -> list[dict]:
    """Match exhibit references in 3.x or 4.x families, common for bylaws/indentures in 8-Ks."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\b(?:Exhibit\s+)?(?:3(?:\.\d+)?|4(?:\.\d+)?)\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
