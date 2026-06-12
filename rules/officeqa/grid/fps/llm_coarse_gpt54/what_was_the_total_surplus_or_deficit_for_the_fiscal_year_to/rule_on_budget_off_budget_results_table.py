def rule_on_budget_off_budget_results_table(doc: dict) -> list[dict]:
    """Match FFO-1 style tables with on-budget/off-budget receipts and outlays plus total surplus/deficit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if (
                re.search(r'on-budget receipts', txt, re.I)
                and re.search(r'off-budget receipts', txt, re.I)
                and re.search(r'total surplus.*deficit', txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
