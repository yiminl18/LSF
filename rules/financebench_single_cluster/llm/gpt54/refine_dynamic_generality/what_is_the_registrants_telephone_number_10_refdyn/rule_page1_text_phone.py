def rule_page1_text_phone(doc: dict) -> list[dict]:
    """Match page-1 text spans that themselves contain the phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "text":
                if phone_re.search((span.get("text") or "")):
                    out.append(span)
        return out
    except Exception:
        return []
