def rule_tables_with_selected_balances_and_public_debt(doc: dict) -> list[dict]:
    """Match tables containing both selected balances and public debt columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'selected balances end of period', text, re.I) and re.search(r'public debt securities', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
