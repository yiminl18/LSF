def rule_exact_fiscal_year_ended_span(doc: dict) -> list[dict]:
    """Match spans whose text directly states 'For the fiscal year ended ...'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the fiscal year ended" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
