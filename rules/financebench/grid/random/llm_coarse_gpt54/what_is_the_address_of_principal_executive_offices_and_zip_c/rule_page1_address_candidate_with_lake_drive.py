def rule_page1_address_candidate_with_lake_drive(doc: dict) -> list[dict]:
    """Match page-1 spans containing Lake Drive address pattern."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\bLake Drive\b', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
