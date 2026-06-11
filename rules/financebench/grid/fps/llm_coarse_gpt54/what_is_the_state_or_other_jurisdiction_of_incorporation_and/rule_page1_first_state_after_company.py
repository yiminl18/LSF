def rule_page1_first_state_after_company(doc: dict) -> list[dict]:
    """Match the first likely state/jurisdiction value on page 1 after the company name block starts."""
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
            if started and re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", (span.get("text") or "").strip(), re.I):
                return [span]
        return []
    except Exception:
        return []
