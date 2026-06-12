def rule_row_fiscal_to_date_in_tables(doc: dict) -> list[dict]:
    """Match table spans containing a row labeled Fiscal ... to date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Fiscal\s+\d{4}\s+to\s+date', txt, re.I) or re.search(r'Actual fiscal year to date', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
