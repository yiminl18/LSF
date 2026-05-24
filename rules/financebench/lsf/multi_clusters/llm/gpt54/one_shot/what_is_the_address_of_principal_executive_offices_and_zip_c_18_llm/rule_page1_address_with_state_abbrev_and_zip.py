def rule_page1_address_with_state_abbrev_and_zip(doc: dict) -> list[dict]:
    """Match page-1 spans containing state abbreviation plus ZIP code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', text):
                out.append(span)
        return out
    except Exception:
        return []
