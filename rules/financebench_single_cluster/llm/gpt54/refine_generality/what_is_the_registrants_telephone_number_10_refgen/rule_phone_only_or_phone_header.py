def rule_phone_only_or_phone_header(doc: dict) -> list[dict]:
    """Match spans whose text is just a phone number or whose header text is just a phone number."""
    import re
    try:
        out = []
        phone_only = re.compile(r"^\s*(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\d{3}-\d{3}-\d{4}|\d{3}-\d{4}-\d{4})\s*$")
        for span in doc.get("texts", []):
            if phone_only.search((span.get("text") or "")):
                out.append(span)
        return out
    except Exception:
        return []
