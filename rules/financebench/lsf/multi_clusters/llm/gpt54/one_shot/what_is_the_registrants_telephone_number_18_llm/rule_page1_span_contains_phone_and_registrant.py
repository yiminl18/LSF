def rule_page1_span_contains_phone_and_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans containing both a phone number and the word registrant."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"registrant", text, re.I) and re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", text):
                out.append(span)
        return out
    except Exception:
        return []
