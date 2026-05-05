def rule_current_report_page1(doc: dict) -> list[dict]:
    """Match page-1 spans with CURRENT REPORT."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and "CURRENT REPORT" in ((span.get("text") or "").upper())
        ]
    except Exception:
        return []
