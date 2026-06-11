def rule_page1_first_15_spans_period(doc: dict) -> list[dict]:
    """Match reporting-period spans among the first 15 page-1 spans."""
    import re
    try:
        page1 = [s for s in doc.get("texts", []) if s.get("page_no") == 1][:15]
        return [
            span for span in page1
            if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|earliest event reported)', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
