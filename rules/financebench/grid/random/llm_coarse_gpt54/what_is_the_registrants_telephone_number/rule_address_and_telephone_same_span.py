def rule_address_and_telephone_same_span(doc: dict) -> list[dict]:
    """Match spans containing 'address and telephone number' language with a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"address and telephone number.*principal executive offices", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
