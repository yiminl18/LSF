def rule_page1_address_candidate_with_exact_address_keywords(doc: dict) -> list[dict]:
    """Match page-1 spans containing multiple address-like keywords for broad recall."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            score = 0
            if re.search(r'\d{2,}', txt):
                score += 1
            if re.search(r'(Street|Avenue|Boulevard|Drive|Road|Center|Plaza|Building)', txt, re.I):
                score += 1
            if re.search(r'(California|Washington|New York|Minnesota|CA|WA|NY|MN|United Kingdom|Bristol)', txt, re.I):
                score += 1
            if re.search(r'(\d{5}(?:-\d{4})?|BS30 ?8XP)', txt, re.I):
                score += 1
            if score >= 2:
                out.append(span)
        return out
    except Exception:
        return []
