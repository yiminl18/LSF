def rule_page1_address_candidate_with_center(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Center' as part of the address."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\bCenter\b', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
