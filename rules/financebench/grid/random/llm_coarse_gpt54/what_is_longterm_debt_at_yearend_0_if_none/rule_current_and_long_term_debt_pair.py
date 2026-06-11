def rule_current_and_long_term_debt_pair(doc: dict) -> list[dict]:
    """Match balance sheet tables containing both current debt and long-term debt rows."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            has_current = re.search(r"current portion of long[\-\s]?term debt|short[\-\s]?term debt|\n\|\s*Debt\s*\|\s*.*\n", text, re.I)
            has_long = re.search(r"long[\-\s]?term debt|long[\-\s]?term liabilities", text, re.I)
            if has_current and has_long:
                out.append(span)
        return out
    except Exception:
        return []
