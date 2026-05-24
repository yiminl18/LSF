def rule_page1_telephone_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing the registrant telephone phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
