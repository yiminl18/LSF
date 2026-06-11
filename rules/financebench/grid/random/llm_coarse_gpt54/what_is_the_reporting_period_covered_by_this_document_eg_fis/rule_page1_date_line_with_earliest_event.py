def rule_page1_date_line_with_earliest_event(doc: dict) -> list[dict]:
    """Match page-1 lines explicitly saying earliest event reported, common for 8-Ks."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'earliest event reported', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
