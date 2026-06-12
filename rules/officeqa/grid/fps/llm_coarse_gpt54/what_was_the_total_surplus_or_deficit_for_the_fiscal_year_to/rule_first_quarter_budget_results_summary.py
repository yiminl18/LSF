def rule_first_quarter_budget_results_summary(doc: dict) -> list[dict]:
    """Match first-quarter budget-results summary tables that restate the fiscal-year-to-date deficit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                re.search(r'budget results for the (first|second|third|fourth) quarter', path, re.I)
                or re.search(r'budget results for the (first|second|third|fourth) quarter', txt, re.I)
            ):
                if re.search(r'total surplus.*deficit', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
