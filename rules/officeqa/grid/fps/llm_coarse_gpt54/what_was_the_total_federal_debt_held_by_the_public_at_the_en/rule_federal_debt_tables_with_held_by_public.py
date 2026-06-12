def rule_federal_debt_tables_with_held_by_public(doc: dict) -> list[dict]:
    """Match any table mentioning 'held by the public', a strong cue for the answer column."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "held by the public" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
