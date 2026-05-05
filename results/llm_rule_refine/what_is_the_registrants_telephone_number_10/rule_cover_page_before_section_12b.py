def rule_cover_page_before_section_12b(doc: dict) -> list[dict]:
    """Match page-1 spans where a phone number appears near the Section 12(b) registration text."""
    import re
    try:
        out = []
        phone_re = r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)"
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and re.search(phone_re + r".{0,150}section 12\(b\)", text, re.I | re.S):
                out.append(span)
        return out
    except Exception:
        return []
