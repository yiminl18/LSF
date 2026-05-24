def rule_page1_large_font_company_header_with_address_in_text_span(doc: dict) -> list[dict]:
    """Match large company header spans whose text_span contains the address block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("size", 0) < 12:
                continue
            text_span = span.get("text_span") or ""
            if re.search(r'\d{1,5} .+', text_span) and (
                re.search(r'\b\d{5}(?:-\d{4})?\b', text_span) or re.search(r'BS30 8XP', text_span, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
