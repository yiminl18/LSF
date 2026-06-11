def rule_phone_span_preceding_telephone_label(doc: dict) -> list[dict]:
    """Match a phone-number span whose next sibling on page 1 is the registrant telephone label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if span.get("page_no") != 1 or nxt.get("page_no") != 1:
                continue
            if phone_re.fullmatch((span.get("text", "") or "").strip()) and "Registrant" in (nxt.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
