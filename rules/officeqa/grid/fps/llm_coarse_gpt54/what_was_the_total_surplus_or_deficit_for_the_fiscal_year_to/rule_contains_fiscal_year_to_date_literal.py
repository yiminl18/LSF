def rule_contains_fiscal_year_to_date_literal(doc: dict) -> list[dict]:
    """Match tables with a literal 'Fiscal year to date' row or column."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'fiscal year to date', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
