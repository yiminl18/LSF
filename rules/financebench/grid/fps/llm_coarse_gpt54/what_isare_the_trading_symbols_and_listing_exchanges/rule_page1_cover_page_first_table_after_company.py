def rule_page1_cover_page_first_table_after_company(doc: dict) -> list[dict]:
    """Match the first page-1 table after the main company header, often the securities registration table."""
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
            if texts[j].get("page_no") == 1 and texts[j].get("label") == "table":
                return [texts[j]]
        return []
    except Exception:
        return []
