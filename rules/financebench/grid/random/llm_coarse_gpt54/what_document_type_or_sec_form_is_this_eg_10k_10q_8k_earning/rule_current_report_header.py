def rule_current_report_header(doc: dict) -> list[dict]:
    """Match CURRENT REPORT section headers, a strong 8-K indicator."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "CURRENT REPORT" in (span.get("text") or "").upper()
        ]
    except Exception:
        return []
