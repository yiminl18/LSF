def rule_page1_any_ended_line(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'ended' plus a date-like year, a broad high-recall rule."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and "ended" in (span.get("text") or "").lower()
            and re.search(r'\b20\d{2}\b', span.get("text") or "")
        ]
    except Exception:
        return []
