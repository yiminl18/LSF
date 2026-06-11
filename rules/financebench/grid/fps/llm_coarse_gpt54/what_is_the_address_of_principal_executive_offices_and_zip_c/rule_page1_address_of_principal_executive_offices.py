def rule_page1_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Match page-1 spans whose text or text_span references address of principal executive offices."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'address of principal executive offices', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
