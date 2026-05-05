def rule_page1_address_of_principal_executive_offices_and_zip_code_8k(doc: dict) -> list[dict]:
    """Match 8-K style page-1 spans with 'address of principal executive offices and zip code' wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices and zip code', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
