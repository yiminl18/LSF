def rule_tables_under_federal_debt_path(doc: dict) -> list[dict]:
    """Match all table spans whose structural path is under Federal Debt."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "federal debt" in (((span.get("structure") or {}).get("path_text") or "").lower())
        ]
    except Exception:
        return []
