def rule_tables_with_selected_balances_end_of_period(doc: dict) -> list[dict]:
    """Match selected-balances end-of-period tables where the answer is often the latest fiscal-year held-by-public value."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "selected balances end of period" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
