def rule_form_and_report_combo_page1(doc: dict) -> list[dict]:
    """Match page-1 spans that together indicate form type by combining FORM heading and report-type language."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        has_form_10k = any(re.fullmatch(r"FORM\s+10-K", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        has_form_10q = any(re.fullmatch(r"FORM\s+10-Q", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        has_form_8k = any(re.fullmatch(r"FORM\s+8-K", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        for span in texts:
            text = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if has_form_10k and re.search(r"ANNUAL REPORT", text, re.I):
                out.append(span)
            if has_form_10q and re.search(r"QUARTERLY REPORT", text, re.I):
                out.append(span)
            if has_form_8k and re.search(r"CURRENT REPORT", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
