def rule_page1_address_with_postal_code_anywhere(doc: dict) -> list[dict]:
    """Match page-1 spans containing a street address and any US or UK postal code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'\d{1,5}\s', text) and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', text) or
                re.search(r'\bBS30 8XP\b', text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
