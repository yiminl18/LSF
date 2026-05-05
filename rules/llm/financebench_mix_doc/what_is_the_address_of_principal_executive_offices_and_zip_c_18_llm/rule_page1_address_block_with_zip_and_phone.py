def rule_page1_address_block_with_zip_and_phone(doc: dict) -> list[dict]:
    """Match page-1 spans whose combined text contains ZIP/postal code and phone number."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if (re.search(r'\b\d{5}(?:-\d{4})?\b', text) or re.search(r'BS30 8XP', text, re.I)) and re.search(r'\(?\+?\d[\d\-\)\( ]{6,}', text):
                out.append(span)
        return out
    except Exception:
        return []
