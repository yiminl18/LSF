def rule_page1_telephone_keyword(doc: dict) -> list[dict]:
    """Match spans on page 1 containing the registrant telephone label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"telephone number.*area code", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
