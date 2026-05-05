def rule_phone_number_answer_extractor_spans(doc: dict) -> list[dict]:
    """Match spans likely to directly contain the answer string itself as a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"^\s*(?:.*?)(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\d{3}-\d{3}-\d{4})(?:.*)\s*$")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
