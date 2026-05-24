def rule_page1_address_block_with_state_and_ein_and_phone(doc: dict) -> list[dict]:
    """Match page-1 spans whose combined text contains state, EIN, address, and phone in one block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'employer identification|i\.r\.s\.', text, re.I) and re.search(r'\d{1,5}\s', text) and re.search(r'\(?\+?\d[\d\-\)\( ]{6,}', text):
                out.append(span)
        return out
    except Exception:
        return []
