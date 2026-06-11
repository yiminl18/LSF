def rule_page1_form_and_company_cluster(doc: dict) -> list[dict]:
    """Match spans in the page-1 form/company header cluster that mention the period."""
    import re
    try:
        spans = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        out = []
        for span in spans[:40]:
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'(FORM\s+10-|CURRENT REPORT|ANNUAL REPORT|QUARTERLY REPORT)', txt, re.I) or re.search(r'(fiscal year ended|quarterly period ended|Date of Report)', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
