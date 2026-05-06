def rule_quarterly_report_on_form_10q_phrase(doc: dict) -> list[dict]:
    """Match spans containing QUARTERLY REPORT ON FORM 10-Q."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "QUARTERLY REPORT ON FORM 10-Q" in ((span.get("text") or "").upper())
            or "QUARTERLY REPORT ON FORM 10-Q" in ((span.get("text_span") or "").upper())
        ]
    except Exception:
        return []
