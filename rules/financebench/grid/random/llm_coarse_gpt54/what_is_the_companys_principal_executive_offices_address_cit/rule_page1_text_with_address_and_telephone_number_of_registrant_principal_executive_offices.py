def rule_page1_text_with_address_and_telephone_number_of_registrant_principal_executive_offices(doc: dict) -> list[dict]:
    """Match page-1 spans containing the combined 'address and telephone number ... principal executive offices' label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'address and telephone number.*principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
