def rule_page1_bold_address_header(doc: dict) -> list[dict]:
    """Match bold page-1 section headers that are the address line immediately associated with principal executive offices."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'address of principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
