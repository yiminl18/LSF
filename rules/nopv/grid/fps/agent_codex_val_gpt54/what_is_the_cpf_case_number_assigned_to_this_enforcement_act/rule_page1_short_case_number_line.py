def rule_page1_short_case_number_line(doc: dict) -> list[dict]:
    """Match short page-1 standalone spans that are just the enforcement case number."""
    try:
        import re

        case_re = re.compile(r"\b(?:CPF\s*)?\d-\d{4}-\d{3}\s*-?\s*NOPV\b", re.IGNORECASE)
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and len((span.get("text") or "").strip()) <= 30
            and case_re.search((span.get("text") or ""))
        ]
    except Exception:
        return []
