def rule_contains_actual_fiscal_year_to_date(doc: dict) -> list[dict]:
    """Match tables with an 'Actual fiscal year to date' column, common in narrative summary pages."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'actual fiscal year to date', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
