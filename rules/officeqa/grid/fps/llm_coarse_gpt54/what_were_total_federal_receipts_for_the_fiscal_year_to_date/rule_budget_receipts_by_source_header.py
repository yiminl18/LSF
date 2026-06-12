def rule_budget_receipts_by_source_header(doc: dict) -> list[dict]:
    """Match section headers explicitly naming Budget Receipts by Source."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r'Budget Receipts by Source', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
