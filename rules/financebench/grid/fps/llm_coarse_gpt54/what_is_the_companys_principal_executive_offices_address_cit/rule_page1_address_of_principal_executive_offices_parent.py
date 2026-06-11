def rule_page1_address_of_principal_executive_offices_parent(doc: dict) -> list[dict]:
    """Match page-1 spans whose own text looks like the address and whose text_span mentions principal executive offices."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            text_span = span.get("text_span") or ""
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices', text_span, re.I):
                if not re.fullmatch(r'\(?address of principal executive offices\)?', text.strip(), re.I):
                    out.append(span)
        return out
    except Exception:
        return []
