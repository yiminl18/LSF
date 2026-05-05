def rule_page1_address_and_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing address text plus telephone label wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'^\s*\d{1,5}\s', span.get("text") or "") and re.search(r'telephone number|area code', text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
