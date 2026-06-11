def rule_telephone_label_following_phone_span(doc: dict) -> list[dict]:
    """Match a phone-number span immediately followed by a telephone-label span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"^\s*(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})\s*$")
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            t1 = span.get("text", "") or ""
            t2 = nxt.get("text", "") or ""
            if span.get("page_no") == 1 and nxt.get("page_no") == 1:
                if phone_re.search(t1) and re.search(r"telephone number.*area code", t2, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
