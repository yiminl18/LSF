def rule_inline_exhibit_sentence_material_agreement(doc: dict) -> list[dict]:
    """Match narrative sentences that identify a material agreement and its exhibit number."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r'(included as|filed as|Exhibit)\s+\d+(\.\d+)?', text, re.I) and re.search(r'agreement|indenture|plan|certificate|deed', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
