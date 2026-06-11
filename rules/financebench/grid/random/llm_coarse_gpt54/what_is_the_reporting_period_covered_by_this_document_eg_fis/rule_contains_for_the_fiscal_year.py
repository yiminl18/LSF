def rule_contains_for_the_fiscal_year(doc: dict) -> list[dict]:
    """Match any span containing 'For the fiscal year'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bFor the fiscal year\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
