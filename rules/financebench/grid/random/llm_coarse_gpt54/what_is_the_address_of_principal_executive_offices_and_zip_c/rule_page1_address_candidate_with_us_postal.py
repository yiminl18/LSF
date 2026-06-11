def rule_page1_address_candidate_with_us_postal(doc: dict) -> list[dict]:
    """Match page-1 spans with US ZIP code patterns in text."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\b\d{5}(?:-\d{4})\b|\b\d{5}\b', (span.get("text") or ""))
        ]
    except Exception:
        return []
