def rule_page1_address_of_principal_executive_offices_or_zip_text_only(doc: dict) -> list[dict]:
    """Match page-1 text-only label spans for address or zip code."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "text"
            and re.search(r'address of principal executive offices|zip code', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
