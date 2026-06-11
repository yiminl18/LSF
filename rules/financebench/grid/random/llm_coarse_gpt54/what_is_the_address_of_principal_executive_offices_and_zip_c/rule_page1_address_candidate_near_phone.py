def rule_page1_address_candidate_near_phone(doc: dict) -> list[dict]:
    """Match page-1 spans near a phone-number span that look like address lines."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'\(\d{3}\)\s*\d{3}[- ]?\d{4}|\+\d{2}', txt):
                for j in range(max(0, i - 4), min(len(texts), i + 1)):
                    s2 = texts[j]
                    if s2.get("page_no") == 1 and re.search(r'\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza|California|Washington|New York|Bristol|United Kingdom|CA|WA)', (s2.get("text") or ""), re.I):
                        out.append(s2)
        return out
    except Exception:
        return []
