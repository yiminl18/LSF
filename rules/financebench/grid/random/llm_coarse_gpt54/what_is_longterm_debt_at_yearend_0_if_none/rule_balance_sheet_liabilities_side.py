def rule_balance_sheet_liabilities_side(doc: dict) -> list[dict]:
    """Match balance sheet tables that include liabilities and debt, narrowing to the liabilities side where the answer lives."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"liabilities", text, re.I) and re.search(r"\bdebt\b|\blong[\-\s]?term debt\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
