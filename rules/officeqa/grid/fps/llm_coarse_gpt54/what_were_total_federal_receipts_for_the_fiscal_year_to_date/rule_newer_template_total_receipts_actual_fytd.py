def rule_newer_template_total_receipts_actual_fytd(doc: dict) -> list[dict]:
    """Match newer-template summary tables with Total receipts and Actual fiscal year to date columns."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and re.search(r'Total receipts', span.get("text") or "", re.I)
            and re.search(r'Actual fiscal year to date', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
