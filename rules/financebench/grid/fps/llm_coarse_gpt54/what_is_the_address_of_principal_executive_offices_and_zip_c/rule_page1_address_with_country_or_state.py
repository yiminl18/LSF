def rule_page1_address_with_country_or_state(doc: dict) -> list[dict]:
    """Match page-1 address spans containing a street plus a state/country/locality token."""
    try:
        import re
        out = []
        states = r'California|New York|New Jersey|Minnesota|Virginia|Washington|Delaware|MD|NJ|WA|VA|CA|MN|United Kingdom|Bristol|San Jose|Cupertino|Arlington|Corning|Issaquah|Bethesda|New Brunswick|Richfield|Warmley|Camden'
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                if re.search(states, t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
