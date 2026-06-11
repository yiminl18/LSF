def rule_bold_phone_only_span(doc: dict) -> list[dict]:
    """Match bold spans that are essentially just the telephone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"^\s*(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|[0-9]{3}[-/][0-9]{3}[-/][0-9]{4})\s*$")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("bold") == 1 and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
