def rule_page1_address_like_section_header_any_level(doc: dict) -> list[dict]:
    """Broadly match any page-1 section_header that looks like a principal office address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', text) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', text) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', text) or
                re.search(r'\bUnited Kingdom\b', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
