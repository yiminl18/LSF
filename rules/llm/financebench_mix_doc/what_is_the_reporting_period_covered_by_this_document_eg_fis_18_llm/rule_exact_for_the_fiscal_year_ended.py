def rule_exact_for_the_fiscal_year_ended(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'For the fiscal year ended'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the fiscal year ended" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
