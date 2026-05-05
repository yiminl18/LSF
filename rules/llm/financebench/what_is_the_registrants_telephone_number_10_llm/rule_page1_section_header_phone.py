def rule_page1_section_header_phone(doc: dict) -> list[dict]:
    """Match page-1 section headers that themselves contain the phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if phone_re.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
