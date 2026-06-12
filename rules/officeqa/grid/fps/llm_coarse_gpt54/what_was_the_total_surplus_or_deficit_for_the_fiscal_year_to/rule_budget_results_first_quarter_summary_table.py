def rule_budget_results_first_quarter_summary_table(doc: dict) -> list[dict]:
    """Match narrative summary tables with 'Actual fiscal year to date' and 'Total surplus (+) or deficit (-)'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if (
                re.search(r'actual fiscal year to date', txt, re.I)
                and re.search(r'total surplus.*deficit', txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
