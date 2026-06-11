def rule_10k_10q_registrant_telephone_parenthetical(doc: dict) -> list[dict]:
    """Match 10-K/10-Q style parenthetical telephone-label spans."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").replace("’", "'")
            if re.search(r"\(Registrant'?s telephone number, including area code\)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
