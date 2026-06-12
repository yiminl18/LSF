def rule_tables_with_end_of_period_and_debt(doc: dict) -> list[dict]:
    """Match tables containing both 'end of period' and debt-related wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'end of period', text, re.I) and re.search(r'debt', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
