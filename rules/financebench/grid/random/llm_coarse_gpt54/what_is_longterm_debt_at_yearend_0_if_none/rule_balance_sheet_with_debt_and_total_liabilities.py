def rule_balance_sheet_with_debt_and_total_liabilities(doc: dict) -> list[dict]:
    """Match balance sheet tables containing both debt and total liabilities, a strong indicator of the right table."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"\btotal liabilities\b", text, re.I) and re.search(r"\bdebt\b|\blong[\-\s]?term debt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
