def rule_tables_with_on_budget_off_budget_and_individual(doc: dict) -> list[dict]:
    """Match later tables that combine on-budget/off-budget framing with individual tax columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'On-Budget|Off-Budget', txt, re.I) and re.search(r'Individual', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
