def rule_page1_case_number_section_header(doc: dict) -> list[dict]:
    """Match page-1 section headers that are exactly the enforcement case number."""
    try:
        import re

        case_re = re.compile(r"\b(?:CPF\s*)?\d-\d{4}-\d{3}\s*-?\s*NOPV\b", re.IGNORECASE)
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "section_header"
            and case_re.search((span.get("text") or ""))
        ]
    except Exception:
        return []
