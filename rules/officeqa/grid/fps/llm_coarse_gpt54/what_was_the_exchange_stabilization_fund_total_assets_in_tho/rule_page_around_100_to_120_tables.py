def rule_page_around_100_to_120_tables(doc: dict) -> list[dict]:
    """Match tables in the 100-120 page range where ESF often appears in 1983-1987 quarterly issues."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and isinstance(span.get("page_no"), int)
            and 100 <= span.get("page_no") <= 120
        ]
    except Exception:
        return []
