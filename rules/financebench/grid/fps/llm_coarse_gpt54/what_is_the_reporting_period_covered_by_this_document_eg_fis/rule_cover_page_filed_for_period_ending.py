def rule_cover_page_filed_for_period_ending(doc: dict) -> list[dict]:
    """Match cover-page spans that say 'Filed ... for the Period Ending ...'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "for the period ending" in (span.get("text") or "").lower()
        ]
    except Exception:
        return []
