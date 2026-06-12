def rule_tables_with_debt_held_by_public_and_debt_subject_to_limit(doc: dict) -> list[dict]:
    """Match Federal Debt tables mentioning both debt held by the public and debt subject to limit/limitation."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and ("held by the public" in (span.get("text") or "").lower() or "debt held by the public" in (span.get("text") or "").lower())
            and ("debt subject to" in (span.get("text") or "").lower() or "statutory limit" in (span.get("text") or "").lower())
        ]
    except Exception:
        return []
