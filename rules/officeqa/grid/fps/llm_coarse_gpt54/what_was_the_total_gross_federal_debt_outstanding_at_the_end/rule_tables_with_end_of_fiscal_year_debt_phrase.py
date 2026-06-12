def rule_tables_with_end_of_fiscal_year_debt_phrase(doc: dict) -> list[dict]:
    """Match tables mentioning end-of-fiscal-year debt concepts explicitly."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'end of (the )?(most recent )?fiscal year', text, re.I) and re.search(r'debt', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
