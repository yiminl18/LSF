def rule_page1_first_phone_number(doc: dict) -> list[dict]:
    """Return the first page-1 span containing a phone-number-like pattern."""
    try:
        import re
        pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and pat.search((span.get("text") or "") + " " + (span.get("text_span") or "")):
                return [span]
        return []
    except Exception:
        return []
