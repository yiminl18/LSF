def rule_tables_with_fiscal_year_or_month_and_held_by_public(doc: dict) -> list[dict]:
    """Match tables whose first column is fiscal year/month and whose last columns include held by the public."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "fiscal year or month" in (span.get("text") or "").lower()
            and "held by the public" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
