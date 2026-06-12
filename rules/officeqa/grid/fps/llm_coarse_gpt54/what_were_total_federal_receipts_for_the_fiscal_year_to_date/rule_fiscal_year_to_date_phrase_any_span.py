def rule_fiscal_year_to_date_phrase_any_span(doc: dict) -> list[dict]:
    """Match any span containing fiscal year to date phrasing."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'(fiscal year to date|actual fiscal year to date|fiscal \d{4} to date)', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
