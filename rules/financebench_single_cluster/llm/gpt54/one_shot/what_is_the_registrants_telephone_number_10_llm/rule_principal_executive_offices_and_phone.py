def rule_principal_executive_offices_and_phone(doc: dict) -> list[dict]:
    """Match spans mentioning principal executive offices together with a phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            low = text.lower()
            if "principal executive offices" in low and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
