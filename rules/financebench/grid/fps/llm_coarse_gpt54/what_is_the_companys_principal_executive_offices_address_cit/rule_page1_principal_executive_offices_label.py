def rule_page1_principal_executive_offices_label(doc: dict) -> list[dict]:
    """Match page-1 spans explicitly labeled as address of principal executive offices."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
