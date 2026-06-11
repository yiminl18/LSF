def rule_transition_report_pursuant(doc: dict) -> list[dict]:
    """Match spans containing TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d)."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(D)" in (span.get("text") or "").upper()
        ]
    except Exception:
        return []
