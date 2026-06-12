def rule_tables_on_pages_15_to_25_with_held_by_public(doc: dict) -> list[dict]:
    """Match mid-document tables on pages 15-25 mentioning held by the public, common in older issues."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and 15 <= int(span.get("page_no", -999)) <= 25
            and "held by the public" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
