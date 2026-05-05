def rule_page1_principal_executive_offices_phrase(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning principal executive offices or executive offices."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'principal executive offices|executive offices', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
