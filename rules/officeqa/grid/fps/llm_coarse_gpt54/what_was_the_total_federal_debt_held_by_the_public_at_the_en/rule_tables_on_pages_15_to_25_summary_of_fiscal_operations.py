def rule_tables_on_pages_15_to_25_summary_of_fiscal_operations(doc: dict) -> list[dict]:
    """Match FFO-1 summary-of-fiscal-operations tables in the common older page range."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and 15 <= int(span.get("page_no", -999)) <= 25
            and "summary of fiscal operations" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
