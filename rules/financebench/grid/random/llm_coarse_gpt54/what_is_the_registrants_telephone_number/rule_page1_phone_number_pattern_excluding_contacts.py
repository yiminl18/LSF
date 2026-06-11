def rule_page1_phone_number_pattern_excluding_contacts(doc: dict) -> list[dict]:
    """Match likely phone-number spans on page 1 while excluding later contact sections."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if span.get("page_no") != 1:
                continue
            if re.search(r"contact|contacts|investor relations", path, re.I):
                continue
            if phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
