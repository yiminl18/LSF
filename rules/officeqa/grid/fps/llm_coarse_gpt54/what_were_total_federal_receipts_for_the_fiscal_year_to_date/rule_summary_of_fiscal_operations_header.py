def rule_summary_of_fiscal_operations_header(doc: dict) -> list[dict]:
    """Match section headers explicitly naming Summary of Fiscal Operations."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r'Summary of Fiscal Operations', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
