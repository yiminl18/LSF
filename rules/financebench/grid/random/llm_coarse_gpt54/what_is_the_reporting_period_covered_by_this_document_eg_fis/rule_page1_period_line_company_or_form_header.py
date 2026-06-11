def rule_page1_period_line_company_or_form_header(doc: dict) -> list[dict]:
    """Match page-1 spans in either form headers or company headers that restate the period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r'(FORM\s+10-|INC\.|CORPORATION|PLC|COMPANY)', path + " " + txt, re.I):
                if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
