def rule_tables_with_summary_of_fiscal_operations_phrase(doc: dict) -> list[dict]:
    """Match any table containing the phrase 'Summary of Fiscal Operations'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "summary of fiscal operations" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
