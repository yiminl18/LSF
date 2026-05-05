def rule_page1_text_span_contains_phone(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span, not just text, contains the phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            ts = span.get("text_span", "") or ""
            if span.get("page_no") == 1 and re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", ts):
                out.append(span)
        return out
    except Exception:
        return []
