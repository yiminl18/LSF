def rule_page1_registrant_telephone_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans explicitly mentioning registrant telephone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"registrant[’'`s]{0,2}\s+telephone number", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
