def rule_page1_address_candidate_with_zip_in_textspan(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span contains a ZIP/postal code and address label context."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'(\d{5}(?:-\d{4})?|BS30 ?8XP)', (span.get("text_span") or ""), re.I)
            and re.search(r'address|zip code', (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
