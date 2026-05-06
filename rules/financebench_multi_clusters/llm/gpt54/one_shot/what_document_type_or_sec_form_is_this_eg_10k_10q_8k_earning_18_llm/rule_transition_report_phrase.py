def rule_transition_report_phrase(doc: dict) -> list[dict]:
    """Match spans containing TRANSITION REPORT PURSUANT..., often adjacent to the true form type."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(D)" in ((span.get("text") or "").upper())
        ]
    except Exception:
        return []
