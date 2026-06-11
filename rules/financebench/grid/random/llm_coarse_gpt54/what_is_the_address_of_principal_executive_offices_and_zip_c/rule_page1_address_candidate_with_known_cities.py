def rule_page1_address_candidate_with_known_cities(doc: dict) -> list[dict]:
    """Match page-1 spans containing known city names seen in this filing family."""
    try:
        import re
        cities = r'\b(?:Seattle|Chicago|San Jose|Santa Monica|Issaquah|Warmley|Bristol|New York|St\. Paul)\b'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and re.search(cities, (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
