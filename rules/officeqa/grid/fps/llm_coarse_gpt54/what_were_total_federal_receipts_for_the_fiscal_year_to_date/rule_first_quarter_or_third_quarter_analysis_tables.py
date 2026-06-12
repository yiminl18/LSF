def rule_first_quarter_or_third_quarter_analysis_tables(doc: dict) -> list[dict]:
    """Match quarter-analysis tables that summarize receipts and fiscal year to date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if span.get("label") == "table" and re.search(r'Total receipts', txt, re.I) and (
                re.search(r'October-December', txt, re.I)
                or re.search(r'Actual fiscal year to date', txt, re.I)
            ):
                out.append(span)
            elif re.search(r'Budget results for the (first|third|fourth) quarter', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
