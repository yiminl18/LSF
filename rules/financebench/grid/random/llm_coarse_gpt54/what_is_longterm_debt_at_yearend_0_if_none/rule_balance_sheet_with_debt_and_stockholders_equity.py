def rule_balance_sheet_with_debt_and_stockholders_equity(doc: dict) -> list[dict]:
    """Match balance sheet tables containing debt and stockholders' equity, another strong balance-sheet signature."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"stockholders.? equity|shareholders.? equity", text, re.I) and re.search(r"\bdebt\b|\blong[\-\s]?term debt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
