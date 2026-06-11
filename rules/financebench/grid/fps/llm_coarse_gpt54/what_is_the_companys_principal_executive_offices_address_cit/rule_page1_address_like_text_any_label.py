def rule_page1_address_like_text_any_label(doc: dict) -> list[dict]:
    """Broadly match any page-1 span whose text looks like a street address plus city/state or city/country."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', text) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?', text) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text) or
                re.search(r'\bUnited Kingdom\b', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
