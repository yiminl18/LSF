def rule_budget_and_off_budget_results_table(doc: dict) -> list[dict]:
    """Match tables headed by 'Budget and off-budget results'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'budget and off-budget results', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
