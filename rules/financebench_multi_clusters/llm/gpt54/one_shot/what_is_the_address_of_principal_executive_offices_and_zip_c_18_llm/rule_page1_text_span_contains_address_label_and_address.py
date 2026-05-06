def rule_page1_text_span_contains_address_label_and_address(doc: dict) -> list[dict]:
    """Match spans where text_span contains the address label and the main text is the address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "")
            if re.search(r'^\d{1,5}\s', text) and re.search(r'address of principal executive offices', text_span, re.I):
                out.append(span)
        return out
    except Exception:
        return []
