def rule_tables_after_federal_debt_header_same_page(doc: dict) -> list[dict]:
    """Match table spans occurring on the same page after a Federal Debt header appears."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header" and "federal debt" in (span.get("text") or "").lower():
                pages.add(span.get("page_no"))
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table" and span.get("page_no") in pages
        ]
    except Exception:
        return []
