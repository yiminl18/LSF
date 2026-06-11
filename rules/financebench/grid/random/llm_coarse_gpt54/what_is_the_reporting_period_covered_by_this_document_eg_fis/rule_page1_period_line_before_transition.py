def rule_page1_period_line_before_transition(doc: dict) -> list[dict]:
    """Match spans where the true period line appears together with the transition-report option."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'(fiscal year ended|quarterly period ended)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
            and re.search(r'TRANSITION REPORT', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
