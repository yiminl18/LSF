def rule_phone_with_registrant_word(doc: dict) -> list[dict]:
    """Match spans containing both a phone number and the word 'registrant'."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if "registrant" in text.lower() and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
