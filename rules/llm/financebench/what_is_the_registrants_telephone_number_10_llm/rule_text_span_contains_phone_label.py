def rule_text_span_contains_phone_label(doc: dict) -> list[dict]:
    """Match spans whose text_span field carries the telephone label or number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            ts = span.get("text_span") or ""
            if span.get("page_no") == 1 and re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number|telephone number, including area code|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d", ts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
