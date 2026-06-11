def rule_item1_business_phone_narrative(doc: dict) -> list[dict]:
    """Match Item 1/Business narrative spans containing a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"Item 1|Business", path, re.I) and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
