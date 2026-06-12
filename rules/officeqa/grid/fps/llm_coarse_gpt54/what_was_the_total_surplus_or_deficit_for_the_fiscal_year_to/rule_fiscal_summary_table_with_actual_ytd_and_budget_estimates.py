def rule_fiscal_summary_table_with_actual_ytd_and_budget_estimates(doc: dict) -> list[dict]:
    """Match summary tables with actual fiscal year to date and budget estimates columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'actual fiscal year to date', txt, re.I) and re.search(r'budget estimates', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
