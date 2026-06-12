def rule_tables_with_gross_federal_debt_outstanding_and_total(doc: dict) -> list[dict]:
    """Match tables containing both 'gross federal debt outstanding' and 'total'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'gross federal debt outstanding', text, re.I) and re.search(r'\btotal\b', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
