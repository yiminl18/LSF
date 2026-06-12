def rule_tables_with_net_budget_receipts_and_individual(doc: dict) -> list[dict]:
    """Match tables that include Net budget receipts and Individual income tax columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Net budget receipts|Net receipts', txt, re.I) and re.search(r'Individual', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
