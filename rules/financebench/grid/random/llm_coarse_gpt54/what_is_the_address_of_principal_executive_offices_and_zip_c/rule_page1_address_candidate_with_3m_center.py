def rule_page1_address_candidate_with_3m_center(doc: dict) -> list[dict]:
    """Match page-1 spans containing 3M Center address pattern."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\b3M Center\b', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
