def rule_cover_outstanding_label(doc: dict) -> list[dict]:
    '''Cover-page (pages 1-3) spans containing "outstanding" — the shares-outstanding label or sentence.'''
    return [
        span for span in doc.get("texts", [])
        if 1 <= span.get("page_no", 0) <= 3
        and "outstanding" in span.get("text", "").lower()
    ]
