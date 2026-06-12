def rule_tables_with_debt_held_by_public_under_financial_operations(doc: dict) -> list[dict]:
    """Match held-by-public tables under Financial Operations, common in later issues."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "held by the public" in (span.get("text") or "").lower()
            and "financial operations" in (((span.get("structure") or {}).get("path_text") or "").lower())
        ]
    except Exception:
        return []
