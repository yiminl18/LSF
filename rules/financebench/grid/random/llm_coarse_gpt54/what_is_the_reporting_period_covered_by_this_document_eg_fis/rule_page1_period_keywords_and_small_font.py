def rule_page1_period_keywords_and_small_font(doc: dict) -> list[dict]:
    """Match small-font page-1 spans with period keywords, often the exact answer line."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and (span.get("size") or 0) <= 10.5
            and re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
