def rule_page1_address_with_country_inline(doc: dict) -> list[dict]:
    """Match page-1 spans where street and city are followed by a country name instead of U.S. state abbreviation."""
    import re
    try:
        out = []
        countries = r'United Kingdom|United States|Canada|Jersey|Australia|Mexico'
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+,?\s+[A-Z][a-z]+.*(?:' + countries + r')', text):
                out.append(span)
        return out
    except Exception:
        return []
