def rule_contains_fiscal_year_ended_any_case(doc: dict) -> list[dict]:
    """Match any span containing 'fiscal year ended' in any capitalization."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'\bfiscal year ended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
