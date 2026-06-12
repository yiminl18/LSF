def rule_row_actual_fiscal_year_to_date(doc: dict) -> list[dict]:
    """Match table spans containing the phrase Actual fiscal year to date."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and re.search(r'Actual fiscal year to date', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
