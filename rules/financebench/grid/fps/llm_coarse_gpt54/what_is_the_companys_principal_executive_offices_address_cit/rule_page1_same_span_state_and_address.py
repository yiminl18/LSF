def rule_page1_same_span_state_and_address(doc: dict) -> list[dict]:
    """Match page-1 spans that combine state of incorporation and principal office address in one text block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1:
                if re.search(r'state or other jurisdiction of incorporation', txt, re.I) and re.search(r'address of principal executive offices', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
