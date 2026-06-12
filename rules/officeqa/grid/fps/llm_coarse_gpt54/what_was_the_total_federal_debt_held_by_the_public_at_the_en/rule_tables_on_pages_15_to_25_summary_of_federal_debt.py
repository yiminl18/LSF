def rule_tables_on_pages_15_to_25_summary_of_federal_debt(doc: dict) -> list[dict]:
    """Match FD-1 summary-of-federal-debt tables in the common older page range."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and 15 <= int(span.get("page_no", -999)) <= 25
            and "summary of federal debt" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
