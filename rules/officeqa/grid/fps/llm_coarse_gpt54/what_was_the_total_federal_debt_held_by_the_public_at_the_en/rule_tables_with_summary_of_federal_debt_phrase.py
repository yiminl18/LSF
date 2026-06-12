def rule_tables_with_summary_of_federal_debt_phrase(doc: dict) -> list[dict]:
    """Match any table containing the phrase 'Summary of Federal Debt'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "summary of federal debt" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
