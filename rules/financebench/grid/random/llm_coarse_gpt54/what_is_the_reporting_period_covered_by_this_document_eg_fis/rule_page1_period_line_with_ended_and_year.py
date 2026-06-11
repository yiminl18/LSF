def rule_page1_period_line_with_ended_and_year(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'ended' and a 4-digit year."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\bended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
            and re.search(r'\b20\d{2}\b', (span.get("text") or "") + " " + (span.get("text_span") or ""))
        ]
    except Exception:
        return []
