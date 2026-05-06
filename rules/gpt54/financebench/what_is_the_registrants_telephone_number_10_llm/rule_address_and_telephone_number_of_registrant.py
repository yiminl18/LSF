def rule_address_and_telephone_number_of_registrant(doc: dict) -> list[dict]:
    """Match spans using the phrase 'address and telephone number ... principal executive offices'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if re.search(r"address\s+and\s+telephone\s+number.*principal executive offices", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
