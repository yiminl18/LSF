def rule_page1_section_header_with_ein_value(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text is an EIN-like value."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
        return out
    except Exception:
        return []
