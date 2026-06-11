def rule_page1_address_and_zip_code_label(doc: dict) -> list[dict]:
    """Match page-1 spans using the combined address-and-zip-code label variant."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'address of principal executive offices and zip code|address of principal executive offices\) \(zip code\)', ((span.get("text") or "") + " " + (span.get("text_span") or "")), re.I)
        ]
    except Exception:
        return []
