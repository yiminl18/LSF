def rule_tables_with_fiscal_year_or_month_and_public_debt(doc: dict) -> list[dict]:
    """Match tables that contain both 'Fiscal year or month' and 'Public debt securities'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'fiscal year or month', text, re.I) and re.search(r'public debt securities', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
