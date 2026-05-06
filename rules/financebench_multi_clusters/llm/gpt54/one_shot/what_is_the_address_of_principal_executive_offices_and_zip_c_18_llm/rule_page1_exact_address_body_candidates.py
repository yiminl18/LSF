def rule_page1_exact_address_body_candidates(doc: dict) -> list[dict]:
    """Match bold page-1 body/text spans that look like a full street-city-state-zip address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            text = (span.get("text") or "").strip()
            if not text:
                continue
            looks_address = (
                re.search(r'\d{1,5} .+,\s*.+,\s*[A-Z][A-Za-z .]+ \d{5}(?:-\d{4})?$', text) or
                re.search(r'\d{1,5} .+\b[A-Z]{2}\b \d{5}(?:-\d{4})?$', text) or
                re.search(r'\d{1,5} .+\bUnited Kingdom\b', text, re.I)
            )
            if looks_address and span.get("bold") == 1:
                out.append(span)
        return out
    except Exception:
        return []
