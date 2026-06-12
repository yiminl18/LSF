def rule_older_template_fiscal_to_date(doc: dict) -> list[dict]:
    """Match older-template tables with row label 'Fiscal YYYY to date'."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table"
            and re.search(r'Fiscal\s+19\d{2}\s+to\s+date|Fiscal\s+20\d{2}\s+to\s+date', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
