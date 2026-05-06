def rule_page1_company_header_children(doc: dict) -> list[dict]:
    """Match body children of the first company-name header on page 1."""
    try:
        spans = doc.get("texts", [])
        company_parent_ids = set()
        for i, span in enumerate(spans):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if any(k in low for k in [
                "amazon.com", "boeing", "costco", "amcor", "corning", "johnson & johnson",
                "lockheed martin", "nike", "ebay"
            ]):
                company_parent_ids.add(i)
        out = []
        for span in spans:
            parent_id = (span.get("structure") or {}).get("parent_id")
            if parent_id in company_parent_ids and span.get("page_no") == 1:
                out.append(span)
        return out
    except Exception:
        return []
