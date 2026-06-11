def rule_page1_period_line_with_date_of(doc: dict) -> list[dict]:
    """Match page-1 spans beginning with 'Date of' and containing report/event wording."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r'^\s*Date of\b', txt, re.I):
                if re.search(r'(Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
