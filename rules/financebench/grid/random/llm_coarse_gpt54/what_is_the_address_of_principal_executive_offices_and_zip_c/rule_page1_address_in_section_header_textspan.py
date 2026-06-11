def rule_page1_address_in_section_header_textspan(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span contains the address label and nearby address content."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "section_header"
            and re.search(r'address of principal executive offices', (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
