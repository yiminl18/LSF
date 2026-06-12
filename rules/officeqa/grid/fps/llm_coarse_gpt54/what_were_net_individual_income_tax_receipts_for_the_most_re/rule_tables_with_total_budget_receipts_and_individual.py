def rule_tables_with_total_budget_receipts_and_individual(doc: dict) -> list[dict]:
    """Match quarter summary tables listing Total budget receipts and Individual income taxes by month."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Total budget receipts', txt, re.I) and re.search(r'Individual income taxes', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
