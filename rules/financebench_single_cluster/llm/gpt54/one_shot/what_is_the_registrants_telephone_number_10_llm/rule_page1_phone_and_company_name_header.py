def rule_page1_phone_and_company_name_header(doc: dict) -> list[dict]:
    """Match page-1 spans where a company-name header block also contains a phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "")
            combo = text + " " + (span.get("text_span") or "")
            if span.get("label") == "section_header" and phone_re.search(combo):
                if text.isupper() or any(tok in text.lower() for tok in ["inc", "corporation", "plc", "company", "wholesale"]):
                    out.append(span)
        return out
    except Exception:
        return []
