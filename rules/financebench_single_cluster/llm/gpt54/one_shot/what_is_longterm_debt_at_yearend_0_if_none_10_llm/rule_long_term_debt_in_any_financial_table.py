def rule_long_term_debt_in_any_financial_table(doc: dict) -> list[dict]:
    """Match any financial-looking table with long-term debt regardless of exact section."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term debt\b", txt):
                out.append(span)
                continue
            if "debt excluding current maturities" in txt:
                out.append(span)
        return out
    except Exception:
        return []
