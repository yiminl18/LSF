def rule_page1_address_and_telephone_number_label(doc: dict) -> list[dict]:
    """Match page-1 spans using the combined address-and-telephone label variant."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'address and telephone number.*principal executive offices', ((span.get("text") or "") + " " + (span.get("text_span") or "")), re.I)
        ]
    except Exception:
        return []
