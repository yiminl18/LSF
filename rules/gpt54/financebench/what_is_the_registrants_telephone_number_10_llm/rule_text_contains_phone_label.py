def rule_text_contains_phone_label(doc: dict) -> list[dict]:
    """Match spans whose text field carries the telephone label or number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            t = span.get("text") or ""
            if span.get("page_no") == 1 and re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number|telephone number, including area code|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d", t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
