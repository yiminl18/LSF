def rule_contains_for_the_fiscal_year_ended_or(doc: dict) -> list[dict]:
    """Match spans with the common 10-K line ending in 'or' after the date."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'For the fiscal year ended .*?\bor\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
