def rule_annual_report_on_form_10k_phrase(doc: dict) -> list[dict]:
    """Match spans containing 'Annual Report on Form 10-K' phrasing."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"Annual Report on Form 10-K", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
