def rule_page1_under_current_report(doc: dict) -> list[dict]:
    """Match page-1 spans under CURRENT REPORT path that mention report dates."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'CURRENT REPORT', path, re.I):
                if re.search(r'(Date of Report|event reported|[A-Z][a-z]+ \d{1,2}, \d{4})', txt):
                    out.append(span)
        return out
    except Exception:
        return []
