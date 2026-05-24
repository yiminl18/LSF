def rule_current_report_heading(doc: dict) -> list[dict]:
    """Match spans with CURRENT REPORT, a common 8-K document-type heading."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "CURRENT REPORT" in ((span.get("text") or "").upper())
        ]
    except Exception:
        return []
