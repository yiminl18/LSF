def rule_tables_on_page_after_federal_debt_header(doc: dict) -> list[dict]:
    """Match tables on pages containing a Federal Debt section header."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if "federal debt" in txt:
                    pages.add(span.get("page_no"))
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table" and span.get("page_no") in pages
        ]
    except Exception:
        return []
