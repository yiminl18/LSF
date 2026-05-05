def rule_page1_small_body_phone(doc: dict) -> list[dict]:
    """Match small-font page-1 body spans containing phone numbers on the cover page."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if (span.get("structure") or {}).get("level") != "Body":
                continue
            if float(span.get("size") or 0) <= 9.5 and phone_re.search((span.get("text") or "")):
                out.append(span)
        return out
    except Exception:
        return []
