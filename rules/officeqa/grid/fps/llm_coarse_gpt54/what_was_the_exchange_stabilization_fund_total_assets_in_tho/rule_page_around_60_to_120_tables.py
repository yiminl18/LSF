def rule_page_around_60_to_120_tables(doc: dict) -> list[dict]:
    """Match tables in the common page range where ESF balance sheets often appear in later bulletins."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and isinstance(span.get("page_no"), int)
            and 60 <= span.get("page_no") <= 120
        ]
    except Exception:
        return []
