def rule_page1_registrant_phone_parenthetical(doc: dict) -> list[dict]:
    """Match spans containing the parenthetical label for registrant telephone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"\(.*registrant[’'`s]{0,2}\s+telephone\s+number.*area code.*\)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
