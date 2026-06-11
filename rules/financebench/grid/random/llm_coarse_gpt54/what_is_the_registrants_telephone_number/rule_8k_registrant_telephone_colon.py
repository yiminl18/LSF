def rule_8k_registrant_telephone_colon(doc: dict) -> list[dict]:
    """Match 8-K style spans containing 'Registrant’s telephone number, including area code:'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").replace("’", "'")
            if re.search(r"Registrant'?s telephone number, including area code\s*:", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
