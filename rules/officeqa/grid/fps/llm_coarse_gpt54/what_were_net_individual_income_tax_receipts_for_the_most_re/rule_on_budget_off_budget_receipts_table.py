def rule_on_budget_off_budget_receipts_table(doc: dict) -> list[dict]:
    """Match later-era FFO-2 tables titled On-Budget and Off-Budget Receipts by Source."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'On-Budget and Off-Budget Receipts by Source', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
