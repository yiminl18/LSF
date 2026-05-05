def rule_page1_text_or_textspan_contains_address_and_telephone(doc: dict) -> list[dict]:
    """Match spans whose text or text_span contains 'address and telephone number'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if span.get("page_no") == 1 and re.search(r"address and telephone number", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
