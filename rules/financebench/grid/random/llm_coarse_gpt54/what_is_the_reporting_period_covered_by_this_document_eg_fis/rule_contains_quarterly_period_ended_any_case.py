def rule_contains_quarterly_period_ended_any_case(doc: dict) -> list[dict]:
    """Match any span containing 'quarterly period ended' in any capitalization."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bquarterly period ended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
