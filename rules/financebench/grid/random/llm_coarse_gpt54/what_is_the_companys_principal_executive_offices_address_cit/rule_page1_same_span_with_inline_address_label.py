def rule_page1_same_span_with_inline_address_label(doc: dict) -> list[dict]:
    """Match page-1 spans whose text or text_span contains the principal executive offices label inline."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'address (and telephone number, including area code, of registrant.?s )?principal executive offices', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
