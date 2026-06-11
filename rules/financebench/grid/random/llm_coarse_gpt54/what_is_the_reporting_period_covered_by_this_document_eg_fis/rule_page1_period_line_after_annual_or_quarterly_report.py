def rule_page1_period_line_after_annual_or_quarterly_report(doc: dict) -> list[dict]:
    """Match spans where annual/quarterly report language and the period appear together."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'(ANNUAL REPORT|QUARTERLY REPORT)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
            and re.search(r'(fiscal year ended|quarterly period ended)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
