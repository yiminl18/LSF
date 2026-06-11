def rule_page1_address_candidate_with_uk_postal(doc: dict) -> list[dict]:
    """Match page-1 spans with UK postal code patterns in text."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
