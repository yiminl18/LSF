def rule_page1_span_with_address_and_phone_inline(doc: dict) -> list[dict]:
    """Match page-1 spans that combine address and phone in one line/block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'\(\d{3}\)\s*\d{3}[-–]\d{4}', txt) and (
                re.search(r'principal executive offices', txt, re.I) or
                re.search(r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+', txt) or
                re.search(r'\b[A-Z]{2}\s+\d{5}', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
