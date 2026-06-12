def rule_tables_with_debt_held_by_public_phrase(doc: dict) -> list[dict]:
    """Match any table containing the exact phrase 'Debt Held by the Public'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "debt held by the public" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
