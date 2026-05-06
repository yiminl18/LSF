def rule_8k_date_of_earliest_event_page1(doc: dict) -> list[dict]:
    """Match page-1 spans with earliest-event date wording for 8-Ks."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and "earliest event reported" in ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
        ]
    except Exception:
        return []
