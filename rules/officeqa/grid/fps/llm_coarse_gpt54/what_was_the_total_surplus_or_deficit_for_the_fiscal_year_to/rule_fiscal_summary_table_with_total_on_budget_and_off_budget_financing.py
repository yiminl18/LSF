def rule_fiscal_summary_table_with_total_on_budget_and_off_budget_financing(doc: dict) -> list[dict]:
    """Match tables containing 'Total on-budget and off-budget financing'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'total on-budget and off-budget financing', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
