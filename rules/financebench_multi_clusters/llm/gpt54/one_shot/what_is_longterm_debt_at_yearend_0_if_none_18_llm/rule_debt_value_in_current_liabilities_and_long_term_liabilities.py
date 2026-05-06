def rule_debt_value_in_current_liabilities_and_long_term_liabilities(doc: dict) -> list[dict]:
    """Match tables where debt appears in both current and long-term liabilities sections."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"current liabilities", txt) and re.search(r"long[- ]term liabilities", txt) and re.search(r"\bdebt\b", txt):
                out.append(span)
    except Exception:
        return []
    return out
