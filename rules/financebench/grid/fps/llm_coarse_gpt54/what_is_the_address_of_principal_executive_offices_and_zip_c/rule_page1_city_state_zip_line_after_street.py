def rule_page1_city_state_zip_line_after_street(doc: dict) -> list[dict]:
    """Match city/state/ZIP lines that immediately follow a street-address line, useful when answer spans are split."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            prev = texts[i-1]
            span = texts[i]
            if span.get("page_no") != 1 or prev.get("page_no") != 1:
                continue
            pt = (prev.get("text") or "").strip()
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', pt, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', pt, re.I):
                if re.search(r'\b[A-Z][a-zA-Z\.\- ]+,\s*[A-Z]{2,}\s+\d{5}(?:-\d{4})?\b', t) or re.search(r'\b[A-Z][A-Za-z ]+,\s*(California|New York|New Jersey|Minnesota|Virginia|Washington|Delaware|Bristol|MD|NJ|WA|VA|CA|MN)\b', t):
                    out.append(span)
        return out
    except Exception:
        return []
