def rule_address_and_telephone_number_principal_offices(doc: dict) -> list[dict]:
    """Match spans containing the combined phrase for address and telephone of principal executive offices."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").replace("’", "'")
            if re.search(r"Address and telephone number, including area code, of registrant'?s principal executive offices", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
