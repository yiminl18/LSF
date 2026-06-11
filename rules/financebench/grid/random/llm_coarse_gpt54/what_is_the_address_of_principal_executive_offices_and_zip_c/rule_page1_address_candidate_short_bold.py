def rule_page1_address_candidate_short_bold(doc: dict) -> list[dict]:
    """Match short bold page-1 spans likely to be address fragments rather than long narrative text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("bold") == 1 and 5 <= len(txt) <= 80:
                if re.search(r'\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza|California|Washington|New York|Bristol|United Kingdom|CA|WA)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
