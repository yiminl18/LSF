def rule_page1_before_securities_registered(doc: dict) -> list[dict]:
    """Match page-1 spans with phone numbers that appear before the securities-registration section."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        cutoff = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and re.search(r"Securities registered pursuant to Section 12\(b\)", span.get("text", "") or "", re.I):
                cutoff = i
                break
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if cutoff is not None and i > cutoff:
                continue
            if phone_re.search(span.get("text", "") or ""):
                out.append(span)
        return out
    except Exception:
        return []
