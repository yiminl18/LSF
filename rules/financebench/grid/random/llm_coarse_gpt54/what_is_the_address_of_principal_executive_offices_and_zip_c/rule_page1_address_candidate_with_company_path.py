def rule_page1_address_candidate_with_company_path(doc: dict) -> list[dict]:
    """Match page-1 spans under a company-name path whose text looks like an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if path and not re.search(r'FORM 10|SECURITIES AND EXCHANGE COMMISSION', path, re.I):
                if re.search(r'\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza|California|Washington|New York|Bristol|United Kingdom|CA|WA)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
