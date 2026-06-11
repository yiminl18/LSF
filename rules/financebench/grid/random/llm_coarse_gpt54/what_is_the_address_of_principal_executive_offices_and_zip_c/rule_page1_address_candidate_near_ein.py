def rule_page1_address_candidate_near_ein(doc: dict) -> list[dict]:
    """Match page-1 spans near an EIN number that look like address lines."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'\b\d{2}-\d{7}\b|\b\d{2,3}-\d{7}\b|\b\d{2}-\d{8}\b', txt):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    s2 = texts[j]
                    if s2.get("page_no") == 1 and re.search(r'\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza|California|Washington|New York|Bristol|United Kingdom|CA|WA)', (s2.get("text") or ""), re.I):
                        out.append(s2)
        return out
    except Exception:
        return []
