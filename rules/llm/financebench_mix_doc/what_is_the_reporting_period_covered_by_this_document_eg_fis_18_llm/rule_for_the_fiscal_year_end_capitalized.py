def rule_for_the_fiscal_year_end_capitalized(doc: dict) -> list[dict]:
    """Match spans using capitalized 'For the Fiscal Year Ended' wording."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the fiscal year ended" in ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
        ]
    except Exception:
        return []
