def rule_annual_report_on_form_10k_phrase(doc: dict) -> list[dict]:
    """Match spans containing ANNUAL REPORT ON FORM 10-K."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "ANNUAL REPORT ON FORM 10-K" in ((span.get("text") or "").upper())
            or "ANNUAL REPORT ON FORM 10-K" in ((span.get("text_span") or "").upper())
        ]
    except Exception:
        return []
