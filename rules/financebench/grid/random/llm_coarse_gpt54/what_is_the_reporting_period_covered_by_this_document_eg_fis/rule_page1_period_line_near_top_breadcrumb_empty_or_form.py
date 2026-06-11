def rule_page1_period_line_near_top_breadcrumb_empty_or_form(doc: dict) -> list[dict]:
    """Match top page-1 spans with empty or FORM-related breadcrumbs and period language."""
    import re
    try:
        out = []
        for span in doc.get("texts", [])[:30]:
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and (path == "" or re.search(r'FORM\s+10-|CURRENT REPORT', path, re.I)):
                if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
