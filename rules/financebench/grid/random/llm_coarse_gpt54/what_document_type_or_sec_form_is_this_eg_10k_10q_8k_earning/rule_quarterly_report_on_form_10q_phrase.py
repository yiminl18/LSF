def rule_quarterly_report_on_form_10q_phrase(doc: dict) -> list[dict]:
    """Match spans containing 'Quarterly Report on Form 10-Q' phrasing."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"Quarterly Report on Form 10-Q", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
