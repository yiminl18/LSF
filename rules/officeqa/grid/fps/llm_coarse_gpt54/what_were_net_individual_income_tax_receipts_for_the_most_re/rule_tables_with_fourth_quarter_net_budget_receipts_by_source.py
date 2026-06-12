def rule_tables_with_fourth_quarter_net_budget_receipts_by_source(doc: dict) -> list[dict]:
    """Match special analysis tables for quarter receipts by source that include Individual income taxes rows."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Net Budget Receipts, by Source', txt, re.I) and re.search(r'Individual income taxes', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
