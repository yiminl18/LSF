def rule_exact_for_the_fiscal_year_ended(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'For the fiscal year ended'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bFor the fiscal year ended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
