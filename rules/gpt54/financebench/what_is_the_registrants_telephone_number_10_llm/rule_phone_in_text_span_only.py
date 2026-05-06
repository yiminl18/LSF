def rule_phone_in_text_span_only(doc: dict) -> list[dict]:
    """Match spans where the phone number appears only in text_span, common in merged cover headers."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if not phone_re.search(span.get("text") or "") and phone_re.search(span.get("text_span") or ""):
                out.append(span)
        return out
    except Exception:
        return []
