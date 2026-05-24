def rule_page1_after_address_label(doc: dict) -> list[dict]:
    """Match page-1 spans where phone text appears near an address label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and re.search(r"address of principal executive offices.{0,150}(telephone|area code|\(\d{3}\))", text, re.I | re.S):
                out.append(span)
        return out
    except Exception:
        return []
