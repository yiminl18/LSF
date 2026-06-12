def rule_tables_with_public_debt_securities_and_agency_securities(doc: dict) -> list[dict]:
    """Match debt balance tables containing public debt securities, agency securities, and government accounts."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and "public debt securities" in (span.get("text") or "").lower()
            and "agency securities" in (span.get("text") or "").lower()
            and "government accounts" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
