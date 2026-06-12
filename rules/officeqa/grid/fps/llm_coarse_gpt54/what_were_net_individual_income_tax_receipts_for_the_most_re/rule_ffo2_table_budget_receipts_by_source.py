def rule_ffo2_table_budget_receipts_by_source(doc: dict) -> list[dict]:
    """Match table spans for Table FFO-2 / Budget Receipts by Source where the answer usually lives."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "table":
                hay = f"{txt}\n{path}"
                if re.search(r'FFO-2', hay, re.I) or re.search(r'Budget Receipts by Source', hay, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
