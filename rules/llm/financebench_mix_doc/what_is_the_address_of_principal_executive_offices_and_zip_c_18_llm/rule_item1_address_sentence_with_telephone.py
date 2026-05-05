def rule_item1_address_sentence_with_telephone(doc: dict) -> list[dict]:
    """Match Item 1/Business spans where address and telephone are stated in prose."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if re.search(r'item 1|business|overview', path, re.I) and re.search(r'located at .*?\d{5}', text, re.I) and re.search(r'telephone number', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
