def rule_phone_in_section_header(doc: dict) -> list[dict]:
    """Match section_header spans that contain a phone number near the registrant identity block."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
