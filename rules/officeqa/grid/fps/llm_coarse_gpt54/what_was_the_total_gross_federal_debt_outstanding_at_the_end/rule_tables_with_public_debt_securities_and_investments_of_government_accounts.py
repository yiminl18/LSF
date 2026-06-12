def rule_tables_with_public_debt_securities_and_investments_of_government_accounts(doc: dict) -> list[dict]:
    """Match tables containing both public debt securities and investments of government accounts."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'public debt securities', text, re.I) and re.search(r'investments? of government accounts', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
