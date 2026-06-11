def rule_principal_corporate_offices_with_phone(doc: dict) -> list[dict]:
    """Match business-overview spans mentioning principal/executive offices and a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"(principal|executive) offices", text, re.I) and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
