def rule_page1_address_like_text_near_company_header(doc: dict) -> list[dict]:
    """Match page-1 text spans near the company header that look like address content."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and txt.upper() == txt and "FORM 10-" not in txt and "SECURITIES AND EXCHANGE" not in txt:
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for j in range(company_idx, min(company_idx + 20, len(texts))):
            txt = (texts[j].get("text") or "").strip()
            if re.search(r'\b\d{5}(?:-\d{4})?\b', txt) or re.search(r'California|Washington|Minnesota|New York|United Kingdom|Bristol', txt, re.I):
                out.append(texts[j])
        return out
    except Exception:
        return []
