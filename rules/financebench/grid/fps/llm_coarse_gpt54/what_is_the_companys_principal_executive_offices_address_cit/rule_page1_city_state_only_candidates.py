def rule_page1_city_state_only_candidates(doc: dict) -> list[dict]:
    """Match page-1 spans that are just the city/state or city/country portion of the principal office address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and (
                re.fullmatch(r'[A-Z][a-z]+,\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?', text) or
                re.fullmatch(r'[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+\d{4,})?', text) or
                re.fullmatch(r'[A-Z][A-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
