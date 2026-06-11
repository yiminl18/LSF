def rule_contains_for_the_transition_period_from_to(doc: dict) -> list[dict]:
    """Match transition-period lines, useful as nearby anchors for the true period line."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bFor the transition period from\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
