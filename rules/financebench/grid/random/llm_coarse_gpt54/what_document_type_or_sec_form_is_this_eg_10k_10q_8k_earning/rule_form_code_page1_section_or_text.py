def rule_form_code_page1_section_or_text(doc: dict) -> list[dict]:
    """Match page-1 section_header/text spans containing a common form code."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") in {"section_header", "text"}
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
