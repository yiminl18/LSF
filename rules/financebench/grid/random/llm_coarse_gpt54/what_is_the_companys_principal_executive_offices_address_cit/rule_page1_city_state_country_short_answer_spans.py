def rule_page1_city_state_country_short_answer_spans(doc: dict) -> list[dict]:
    """Match short page-1 spans that are just city/state or city/country answers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.fullmatch(r'[A-Z][A-Za-z .\'-]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York)', txt):
                out.append(span)
            elif re.fullmatch(r'[A-Z][A-Za-z .\'-]+,\s*[A-Z][A-Za-z .\'-]+,\s*United Kingdom', txt):
                out.append(span)
        return out
    except Exception:
        return []
