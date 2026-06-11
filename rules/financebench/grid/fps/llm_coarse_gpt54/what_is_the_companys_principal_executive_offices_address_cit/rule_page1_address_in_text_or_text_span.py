def rule_page1_address_in_text_or_text_span(doc: dict) -> list[dict]:
    """Broadly match any page-1 span whose text or text_span contains a street address plus city/state."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'\d{1,6}\s+\S+.*(?:[A-Z][a-z]+,\s*[A-Z]{2}|[A-Z][a-z]+,\s*[A-Z][a-z]+|United Kingdom)', txt):
                out.append(span)
        return out
    except Exception:
        return []
