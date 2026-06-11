def rule_form_code_page1_bold_only(doc: dict) -> list[dict]:
    """Match bold page-1 spans containing a common form code."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("bold") == 1
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
