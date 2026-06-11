def rule_page1_principal_executive_offices_phrase(doc: dict) -> list[dict]:
    """Match page-1 spans containing the phrase principal executive offices."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'principal executive offices', ((span.get("text") or "") + " " + (span.get("text_span") or "")), re.I)
        ]
    except Exception:
        return []
