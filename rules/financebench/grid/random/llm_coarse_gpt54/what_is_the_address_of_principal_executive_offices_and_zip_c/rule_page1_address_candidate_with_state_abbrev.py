def rule_page1_address_candidate_with_state_abbrev(doc: dict) -> list[dict]:
    """Match page-1 spans containing common state abbreviations near address text."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r'\b(?:CA|WA|NY|MN|IL)\b', (span.get("text") or ""))
            and re.search(r'\d{2,}|Street|Avenue|Boulevard|Drive|Road|Center|Plaza', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
