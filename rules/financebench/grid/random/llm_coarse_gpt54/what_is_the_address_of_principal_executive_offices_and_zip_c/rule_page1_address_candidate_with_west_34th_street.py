def rule_page1_address_candidate_with_west_34th_street(doc: dict) -> list[dict]:
    """Match page-1 spans containing West 34th Street address pattern."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'West 34th Street', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
