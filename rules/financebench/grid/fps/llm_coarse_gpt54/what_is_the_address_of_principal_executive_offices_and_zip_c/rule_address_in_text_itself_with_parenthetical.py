def rule_address_in_text_itself_with_parenthetical(doc: dict) -> list[dict]:
    """Match spans whose text itself contains both an address and the office-address parenthetical."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            t = span.get("text") or ""
            if re.search(r'address of principal executive offices', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
