def rule_page1_address_candidate_with_hamilton_avenue(doc: dict) -> list[dict]:
    """Match page-1 spans containing Hamilton Avenue address pattern."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\bHamilton Avenue\b', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
