def rule_page1_cover_page_first_exchange_after_company(doc: dict) -> list[dict]:
    """Match the first page-1 exchange-name span after the main company header."""
    import re
    try:
        texts = doc.get("texts", [])
        company_idx = None
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and s.get("label") == "section_header":
                txt = (s.get("text") or "")
                if "FORM 8-K" not in txt and "FORM 10-K" not in txt and "FORM 10-Q" not in txt:
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for j in range(company_idx, min(len(texts), company_idx + 25)):
            if texts[j].get("page_no") == 1 and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", (texts[j].get("text") or ""), re.I):
                return [texts[j]]
        return []
    except Exception:
        return []
