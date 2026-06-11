def rule_page1_text_contains_address_and_phone(doc: dict) -> list[dict]:
    """Match page-1 spans that combine address and phone in one line."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            if re.search(r'\d{2,}.*\(\d{3}\)|\d{2,}.*\+\d{2}', txt):
                out.append(span)
        return out
    except Exception:
        return []
