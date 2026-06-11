def rule_page1_section_header_with_ended(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text or text_span contains 'ended'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "section_header"
            and re.search(r'\bended\b', (span.get("text") or "") + " " + (span.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
