def rule_phone_before_section12b(doc: dict) -> list[dict]:
    """Match spans with phone numbers that occur before the Section 12(b) securities registration block."""
    try:
        import re
        texts = doc.get("texts", [])
        sec_idx = None
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"section 12\(b\)", text, re.I):
                sec_idx = i
                break
        if sec_idx is None:
            sec_idx = len(texts)
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in texts[:sec_idx]:
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
