def rule_exhibit_99_or_104_when_no_material_agreement(doc: dict) -> list[dict]:
    """Match exhibit references to 99.x or 104, useful for documents whose answer is a non-10 exhibit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\b(?:Exhibit\s+)?(?:99(?:\.\d+)?|104)\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
