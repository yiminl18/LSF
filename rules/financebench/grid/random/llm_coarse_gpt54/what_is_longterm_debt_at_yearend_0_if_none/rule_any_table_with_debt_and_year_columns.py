def rule_any_table_with_debt_and_year_columns(doc: dict) -> list[dict]:
    """Match debt tables that also have year/date columns, common for balance sheet answer rows."""
    try:
        import re
        out = []
        year_pat = r"\b(20\d{2}|19\d{2}|june|december|november|dec\.?|jun\.?)\b"
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"\bdebt\b|\blong[\-\s]?term debt\b", text, re.I) and re.search(year_pat, text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
