def rule_page1_text_with_city_state_pattern(doc: dict) -> list[dict]:
    """Match page-1 spans containing a city/state or city/country pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\b[A-Z][A-Za-z.\' -]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York|United Kingdom)\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
