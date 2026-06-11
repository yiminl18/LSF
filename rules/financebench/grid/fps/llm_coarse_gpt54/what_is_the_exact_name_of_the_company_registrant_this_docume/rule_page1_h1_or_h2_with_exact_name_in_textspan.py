def rule_page1_h1_or_h2_with_exact_name_in_textspan(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span itself contains the exact-name annotation."""
    try:
        texts = doc.get("texts", [])
        return [
            span for span in texts
            if span.get("page_no") == 1
            and span.get("label") == "section_header"
            and "exact name of registrant as specified in its charter" in (span.get("text_span", "") or "").lower()
        ]
    except Exception:
        return []
