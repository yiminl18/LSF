def rule_page1_text_with_address_of_principal_executive_offices_in_textspan(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span contains the principal executive offices label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text_span") or "")
            if re.search(r'address of principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
