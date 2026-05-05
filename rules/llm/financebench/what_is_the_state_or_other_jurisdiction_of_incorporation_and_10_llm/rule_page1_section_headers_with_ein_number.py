def rule_page1_section_headers_with_ein_number(doc: dict) -> list[dict]:
    """Match page-1 section headers containing an EIN number pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and span.get("label") == "section_header" and re.search(r"\d{2}-\d{7}", text):
                out.append(span)
        return out
    except Exception:
        return []
