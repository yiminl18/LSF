def rule_page1_after_address_before_securities(doc: dict) -> list[dict]:
    """Match spans between address label and securities-registration label on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and start is None and re.search(r"address of principal executive offices|address and telephone number", (span.get("text", "") or "") + " " + (span.get("text_span", "") or ""), re.I):
                start = i
            if span.get("page_no") == 1 and end is None and re.search(r"securities registered pursuant to section 12\(b\)", span.get("text", "") or "", re.I):
                end = i
                break
        if start is None:
            start = 0
        if end is None:
            end = len(texts)
        out = []
        for span in texts[start:end]:
            if span.get("page_no") != 1:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
