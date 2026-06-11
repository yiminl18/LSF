def rule_current_report_header_line(doc: dict) -> list[dict]:
    """Match CURRENT REPORT headers on page 1 for 8-Ks."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and "current report" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
