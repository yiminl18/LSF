def rule_page1_case_number_near_top(doc: dict) -> list[dict]:
    """Match page-1 case-number spans that appear in the opening block of the letter."""
    try:
        import re

        case_re = re.compile(r"\b(?:CPF\s*)?\d-\d{4}-\d{3}\s*-?\s*NOPV\b", re.IGNORECASE)
        return [
            span for i, span in enumerate(doc.get("texts", []))
            if span.get("page_no") == 1
            and i <= 15
            and case_re.search((span.get("text") or ""))
        ]
    except Exception:
        return []
