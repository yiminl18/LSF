def rule_page1_period_keywords_in_text_span(doc: dict) -> list[dict]:
    """Match spans whose text_span, rather than text, contains the reporting period."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', span.get("text_span") or "", re.I)
        ]
    except Exception:
        return []
