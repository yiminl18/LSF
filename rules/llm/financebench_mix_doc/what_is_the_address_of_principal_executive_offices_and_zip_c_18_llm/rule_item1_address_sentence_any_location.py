def rule_item1_address_sentence_any_location(doc: dict) -> list[dict]:
    """Match Item 1/Business prose spans containing a full address pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if re.search(r'item 1|business|overview', path, re.I):
                if re.search(r'\d{1,5} .+,\s*.+,\s*[A-Z][A-Za-z .]+ \d{5}(?:-\d{4})?', text) or re.search(r'\d{1,5} .+\bUnited Kingdom\b', text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
