def rule_deferred_revenue_not_debt_exclusion_balance_sheet(doc: dict) -> list[dict]:
    """Match balance sheet tables with debt while excluding common false-positive deferred revenue tables."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"deferred revenue", text, re.I) and not re.search(r"balance sheet", text, re.I):
                continue
            if re.search(r"balance sheet", text, re.I) and re.search(r"\bdebt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
