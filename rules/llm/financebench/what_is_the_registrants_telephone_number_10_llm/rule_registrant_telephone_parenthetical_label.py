def rule_registrant_telephone_parenthetical_label(doc: dict) -> list[dict]:
    """Match label spans that explicitly say '(Registrant’s telephone number, including area code)'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number,\s*including\s+area\s+code", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
