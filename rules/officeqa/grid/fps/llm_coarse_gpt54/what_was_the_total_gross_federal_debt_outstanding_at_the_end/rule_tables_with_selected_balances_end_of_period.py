def rule_tables_with_selected_balances_end_of_period(doc: dict) -> list[dict]:
    """Match tables containing 'Selected balances end of period', which often include the year-end debt figure."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'selected balances end of period', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
