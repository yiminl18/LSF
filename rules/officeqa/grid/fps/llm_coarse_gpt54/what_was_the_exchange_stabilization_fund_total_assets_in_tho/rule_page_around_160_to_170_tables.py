def rule_page_around_160_to_170_tables(doc: dict) -> list[dict]:
    """Match tables in the 160-170 page range where ESF appears in some 1984-1985 issues."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and isinstance(span.get("page_no"), int)
            and 160 <= span.get("page_no") <= 170
        ]
    except Exception:
        return []
