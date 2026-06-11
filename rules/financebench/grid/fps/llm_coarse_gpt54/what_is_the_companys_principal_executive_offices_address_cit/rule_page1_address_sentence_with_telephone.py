def rule_page1_address_sentence_with_telephone(doc: dict) -> list[dict]:
    """Match narrative spans that mention executive offices and telephone number together."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'executive offices.*telephone number', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
