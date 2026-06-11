def rule_registrant_telephone_exact_phrase(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'Registrant’s telephone number' or close variants."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").replace("’", "'")
            if re.search(r"Registrant'?s telephone number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
