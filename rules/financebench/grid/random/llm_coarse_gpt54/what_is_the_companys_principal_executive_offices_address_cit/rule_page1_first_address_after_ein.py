def rule_page1_first_address_after_ein(doc: dict) -> list[dict]:
    """Return the first address-like span after an EIN marker on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        seen_ein = False
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'Employer Identification No|I\.R\.S\.', txt, re.I):
                seen_ein = True
                continue
            if seen_ein and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', txt) or
                re.search(r'\b[A-Z][A-Za-z .-]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York|United Kingdom)\b', txt)
            ):
                out.append(span)
                break
        return out
    except Exception:
        return []
