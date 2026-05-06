def rule_cover_page_phone_with_zip(doc: dict) -> list[dict]:
    """Match cover-page spans containing both zip-code wording and a phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and phone_re.search(text) and "zip code" in text.lower():
                out.append(span)
        return out
    except Exception:
        return []
