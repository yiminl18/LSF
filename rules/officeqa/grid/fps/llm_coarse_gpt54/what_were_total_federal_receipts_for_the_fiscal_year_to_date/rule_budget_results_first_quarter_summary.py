def rule_budget_results_first_quarter_summary(doc: dict) -> list[dict]:
    """Match first-quarter budget results summary sections/tables that state total receipts and fiscal year to date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                re.search(r'Budget results for the first quarter', txt, re.I)
                or re.search(r'Budget results for the first quarter', path, re.I)
                or re.search(r'first quarter, fiscal', txt, re.I)
            ):
                out.append(span)
            elif span.get("label") == "table" and re.search(r'Total receipts', txt, re.I) and re.search(r'Actual fiscal year to date', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
