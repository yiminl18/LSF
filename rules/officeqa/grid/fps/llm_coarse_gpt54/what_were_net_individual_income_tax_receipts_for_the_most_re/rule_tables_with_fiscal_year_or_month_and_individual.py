def rule_tables_with_fiscal_year_or_month_and_individual(doc: dict) -> list[dict]:
    """Match detailed FFO-2 tables with a Fiscal year or month stub and individual tax columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Fiscal year or month', txt, re.I) and re.search(r'Individual', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
