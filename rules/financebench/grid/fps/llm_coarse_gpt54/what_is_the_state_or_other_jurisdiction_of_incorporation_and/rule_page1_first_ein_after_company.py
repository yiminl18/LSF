def rule_page1_first_ein_after_company(doc: dict) -> list[dict]:
    """Match the first EIN-like span on page 1 after the company name block starts."""
    try:
        import re
        texts = doc.get("texts", [])
        started = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if not started and re.search(r"exact name of registrant", txt, re.I):
                started = True
            if started and re.fullmatch(r"\d{2}-\d{7}", (span.get("text") or "").strip()):
                return [span]
        return []
    except Exception:
        return []
