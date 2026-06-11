def rule_page1_text_contains_address_and_city(doc: dict) -> list[dict]:
    """Match page-1 spans with street number and city-like tokens."""
    try:
        import re
        city_words = r'(Seattle|Chicago|San Jose|Santa Monica|Issaquah|Warmley|Bristol|New York|St\. Paul)'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\d{2,}.*' + city_words, (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
