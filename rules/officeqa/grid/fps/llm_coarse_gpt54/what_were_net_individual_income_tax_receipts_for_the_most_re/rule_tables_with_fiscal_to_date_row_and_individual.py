def rule_tables_with_fiscal_to_date_row_and_individual(doc: dict) -> list[dict]:
    """Match FFO-2 tables containing a Fiscal-to-date row and individual income tax columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Fiscal\s+\d{4}\s+to\s+date|Fiscal\s+to\s+date', txt, re.I) and re.search(r'Individual', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
