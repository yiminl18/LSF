def rule_debt_table_with_liabilities_and_equity_context(doc: dict) -> list[dict]:
    """Match balance-sheet style tables containing debt plus liabilities/equity context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                (re.search(r"\blong[\-\s]?term debt\b", txt) or "debt excluding current maturities" in txt)
                and ("shareholders" in txt or "stockholders" in txt or "equity" in txt or "liabilities" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
