def rule_page1_phone_number_pattern_near_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans containing a phone-number pattern and registrant/telephone wording."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\d{3}[-]\d{3}[-]\d{4})")
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and phone_re.search(text) and re.search(r"(registrant|telephone|area code)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
