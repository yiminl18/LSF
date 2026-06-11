def rule_page1_period_keywords_and_bold(doc: dict) -> list[dict]:
    """Match bold page-1 spans with period keywords, regardless of label."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("bold") == 1
            and re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
