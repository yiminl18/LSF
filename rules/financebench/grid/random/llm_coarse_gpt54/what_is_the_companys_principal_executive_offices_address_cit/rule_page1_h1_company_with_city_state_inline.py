def rule_page1_h1_company_with_city_state_inline(doc: dict) -> list[dict]:
    """Match company H1 section headers on page 1 whose text_span includes city/state or full address inline."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text_span") or "")
            if re.search(r'\b[A-Z][A-Za-z .-]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York|United Kingdom)\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
