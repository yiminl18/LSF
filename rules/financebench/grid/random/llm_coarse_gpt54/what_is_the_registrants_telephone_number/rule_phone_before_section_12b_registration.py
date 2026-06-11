def rule_phone_before_section_12b_registration(doc: dict) -> list[dict]:
    """Match phone-number spans appearing before the Section 12(b) securities registration block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        first_sec12b = None
        for i, span in enumerate(texts):
            if re.search(r"Section 12\(b\)", span.get("text", "") or "", re.I):
                first_sec12b = i
                break
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            if first_sec12b is not None and i > first_sec12b:
                continue
            if span.get("page_no") == 1 and phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
