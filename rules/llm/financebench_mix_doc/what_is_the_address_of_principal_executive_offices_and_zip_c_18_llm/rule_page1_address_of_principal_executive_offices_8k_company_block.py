def rule_page1_address_of_principal_executive_offices_8k_company_block(doc: dict) -> list[dict]:
    """Match 8-K company block spans where address and zip are embedded in the same large header."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'address of principal executive offices', text, re.I) and re.search(r'\b\d{5}(?:-\d{4})?\b', text):
                out.append(span)
        return out
    except Exception:
        return []
