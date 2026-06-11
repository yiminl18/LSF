def rule_telephone_label_with_number_same_span(doc: dict) -> list[dict]:
    """Match spans where the telephone label and the number appear in the same text span."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"telephone number", text, re.I) and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
