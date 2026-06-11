def rule_apple_balance_sheet_long_term_debt_table(doc: dict) -> list[dict]:
    """Match Apple-style balance sheet tables likely containing long-term debt in millions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if "apple inc." in path and "balance" in path:
                out.append(span)
                continue
            text = (span.get("text") or "").lower()
            if "long-term debt" in text and "total liabilities" in text:
                out.append(span)
        return out
    except Exception:
        return []
