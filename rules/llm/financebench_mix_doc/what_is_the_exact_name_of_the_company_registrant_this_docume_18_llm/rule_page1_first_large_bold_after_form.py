def rule_page1_first_large_bold_after_form(doc: dict) -> list[dict]:
    """Match the first large bold page-1 span after a FORM header that is likely the registrant name."""
    try:
        import re
        texts = doc.get("texts", [])
        form_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and re.search(r"FORM 10-|FORM 8-K", span.get("text") or "", re.I):
                form_idx = i
                break
        if form_idx is None:
            return []
        for j in range(form_idx + 1, len(texts)):
            span = texts[j]
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                break
            if (span.get("size") or 0) >= 10 and span.get("bold") == 1 and len(txt.split()) >= 2:
                if "CURRENT REPORT" not in txt.upper() and "COMMISSION" not in txt.upper() and "FORM" not in txt.upper():
                    return [span]
        return []
    except Exception:
        return []
