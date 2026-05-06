def rule_page1_bold_small_text_identifiers(doc: dict) -> list[dict]:
    """Match small bold page-1 text spans that often hold split cover-page identifiers."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            size = span.get("size", 0) or 0
            if span.get("page_no") == 1 and span.get("bold") == 1 and size <= 10.5 and text:
                out.append(span)
        return out
    except Exception:
        return []
