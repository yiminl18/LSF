def rule_page1_period_and_form_type_same_span(doc: dict) -> list[dict]:
    """Match spans containing both the form type language and the period in one span."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'(ANNUAL REPORT|QUARTERLY REPORT|CURRENT REPORT)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
            and re.search(r'(fiscal year ended|quarterly period ended|Date of Report)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
