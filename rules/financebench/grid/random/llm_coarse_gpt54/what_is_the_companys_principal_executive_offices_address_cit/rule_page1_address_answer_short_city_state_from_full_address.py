def rule_page1_address_answer_short_city_state_from_full_address(doc: dict) -> list[dict]:
    """Match page-1 full-address spans from which the city/state answer can be derived."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\b\d+\s+.*(?:Seattle,\s*Washington|San Jose,\s*California|Issaquah,\s*WA|Santa Monica,\s*CA|St\. Paul,\s*Minnesota|New York,\s*New York|Chicago,\s*IL|Warmley,\s*Bristol.*United Kingdom)', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
